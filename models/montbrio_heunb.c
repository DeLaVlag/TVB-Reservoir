#include <stdio.h> // for printf
#define PI_2 (2 * M_PI_F)
#define PI M_PI_F
#define INF INFINITY

// buffer length defaults to the argument to the integrate kernel
// but if it's known at compile time, it can be provided which allows
// compiler to change i%n to i&(n-1) if n is a power of two.
#ifndef NH
#define NH nh
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#include <curand_kernel.h>
#include <curand.h>
#include <stdbool.h>

__device__ float wrap_it_r(float r)
{
    float rdim[] = {0.0, inf};
    if (r < rdim[0]) r = rdim[0];
    else if (r > rdim[1]) r = rdim[1];

    return r;
}

__device__ float wrap_it_V(float V)
{
    float Vdim[] = {-100., 100.};
    if (V < Vdim[0]) V = Vdim[0];
    else if (V > Vdim[1]) V = Vdim[1];

    return V;
}

__global__ void montbrio_heun(

        unsigned int i_step, unsigned int n_node, unsigned int nh, unsigned int n_step, unsigned int n_work_items,
        float dt, float conduct_speed, int my_rank, int inject_dyn, int buffoffzet, int thoughtsize, int train_or_pred,
//        float * __restrict__ weights,
        float * __restrict__ weights_pwi,
        float * __restrict__ lengths,
        float * __restrict__ params_pwi, // pwi: per work item
        // state
        float * __restrict__ state_pwi,
        // outputs
        float * __restrict__ tavg_pwi,
        float * __restrict__ x,
        float * __restrict__ y,
        float * __restrict__ z,
        float * __restrict__ xt,
        float * __restrict__ yt,
        float * __restrict__ zt,
        float * __restrict__ thoughts_pwi
        )
{
    // work id & size
//    const unsigned int id = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int id =
    (blockIdx.y * gridDim.x * blockDim.x * blockDim.y) + // Offset from grid rows
    (blockIdx.x * blockDim.x * blockDim.y) +            // Offset from grid columns
    (threadIdx.y * blockDim.x) +                        // Offset from block rows
    threadIdx.x;                                        // Offset from block columns

    const unsigned int size = n_work_items;

    // only threat those ID that have a corresponding parameters combination
    if (id >= size) return;

    float *weights = weights_pwi + (id * n_node * n_node);
    float *thougths = thoughts_pwi + (id * thoughtsize * 3);

#define params(i_par) (params_pwi[(size * (i_par)) + id])
#define state(time, i_node) (state_pwi[((time) * 2 * n_node + (i_node))*size + id])
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])

        //if (my_rank == 0 && id==0 && i_step==0)
            //printf("size %d\n", size);


    // unpack params
    // These are the two parameters which are usually explore in fitting in this model
    const float global_coupling = params(0);
    const float nsig = params(1);
    const float I = params(2);
    const float J = params(3);
    const float eta = params(4);
    const float Delta = params(5);
    const float global_speed = params(6);



    // regular constants
    const float tau = 1.0;   //Characteristic time
//    const float I = 0.0;     //External Current
//    const float Delta = 0.7; //Mean heterogeneous noise
//    const float J = 14.5;    //Mean Synaptic weight
//    const float eta = -4.6;  //Constant parameter to scale the rate of feedback from the firing rate variable to itself
    const float Gamma = 5.0; //Half-width of synaptic weight distribution
    const float cr = 1.0;    //weight on Coupling through variable r
    const float cv = 0;      //weight on Coupling through variable V

//    const float global_speed = 4;      //weight on Coupling through variable V

    // coupling constants, coupling itself is hardcoded in kernel

    // coupling parameters
    float c_pop0 = 0.0;
    float c_pop1 = 0.0;

    // derived parameters
    const float rec_n = 1 / n_node;
    const float rec_speed_dt = 1.0f / global_speed / dt;
//    const float rec_speed_dt = 1.0f / 1.0f / dt;
    float nsig_r = 0.01;
    float nsig_V = 0.02;
//    const float nsig = 0.01;

    curandState crndst;
//    curand_init(42, 0, 0, &crndst);
    curand_init(id + (unsigned int) clock64(), 0, 0, &crndst);

    float noise_r = 0.0;
    float noise_V = 0.0;
    float sqrtdt = 0.0;
    float sqrtnsig = 0.0;

    float r = 0.0;
    float V = 0.0;
    float dr_i = 0.0;
    float dV_i = 0.0;
    float dr = 0.0;
    float dV = 0.0;

    unsigned int dij_i = 0;
    float dij = 0.0;
    float wij = 0.0;

    float r_j = 0;
    float V_j = 0;

    int euler_integrated = 1;
    int heun_integrated = 0;

        // stiumulus variables in timesteps

    int onset = 25000;
    int duration = 1500;
    // encoding info in the membrane which is negative.
    // weights closer to 0 mean permanent stimulus
    // and thus firing rate is enabled via noise
//    when not normalized good setting:

    // for envoded stuff
//    float weight = 1;
//    float offsetab = -.5;
//    float offsetc = -.5;



    // for the sinusoidials
//    float weight = 2;
//    float offsetab = -.4;
//    float offsetc = -.4;
//    float reststater = .25;

        // for r injection
    float weight = 1;
    float offsetab = 0;
    float offsetc = 0;
    float reststater = .5;
    int teacherdelay = 225; // to match phase shift in prediction
//    int teacherdelay = 100;

    // for the inverted R trials
//    float weight = 3e-2;
//    float offsetab = -.5;
//    float offsetc = -1;


    // non inverted for lorentz latest setting!
//    float weight = 3e-2;
//    float offsetab = -.10;
//    float offsetc = -.8;
//    float reststater = .75;

//    printf("offsetab %f\n", offsetab);
//    printf("offsetc %f\n", offsetc);

    // tryout when normalize
//    float weight = 3;
//    float offset = 6;

    int printweight = 0;
    if (printweight == 1)
    {
        if (i_step == 0 && id == 0)
        {
            printf("\n");
            for (unsigned int i_node = 0; i_node < n_node; i_node++)
            {
                for (unsigned int j_node = 0; j_node < n_node; j_node++)
                {
                    if (j_node == 23)// || j_node == 16 || j_node == 21)
                    {
                        float pwij = weights[i_node * n_node + j_node];
                        printf("in %d jn %d w %.2e\n", i_node, j_node, pwij);
                    }
                }
            }
        }
    }

    int printthoughts = 0;
    if (printthoughts == 1)
    {
        if (i_step == 0 && id == 0)
        {
            printf("\n");
            for (unsigned int tidx = 0; tidx < 100; tidx++)
             {
                    printf("contentss %f\n", thougths[tidx+thoughtsize*0]);

             }
        }
        return;
    }

    int printparams = 0;
    if (printparams == 1)
    {
        // Print parameters
        if (my_rank == 0 && id==0 && i_step==0)
        {
            printf("   INFO  Model parameters:\n");
            printf("   INFO  global_coupling =      %.1f \n", global_coupling);
            printf("   INFO  global speed =         %.1f \n", global_speed);
            printf("   INFO  n_sig =                %.6f \n", nsig);
            printf("   INFO  tau =                  %.1f \n", tau);
            printf("   INFO  I =                    %.1f \n", I);
            printf("   INFO  Delta =                %.1f \n", Delta);
            printf("   INFO  J =                    %.1f \n", J);
            printf("   INFO  eta =                  %.1f \n", eta);
            printf("   INFO  Gamma =                %.1f \n", Gamma);
            printf("   INFO  cr =                   %.1f \n", cr);
            printf("   INFO  cv =                   %.1f \n", cv);
            printf("   INFO  rec_speed_dt =         %.1f \n", rec_speed_dt);
            printf("   INFO  stim onset =           %d \n", onset);
            printf("   INFO  stim duration =        %d \n", duration);
            printf("   INFO  stim weight =          %.1f \n", weight);
            printf("\n\n");

        }
    }

    //***// This is only initialization of the observable
    for (unsigned int i_node = 0; i_node < n_node; i_node++)
    {
        tavg(i_node) = 0.0f;
        if (i_step == 0){
            state(i_step, i_node) = 0.0f;
        }
    }


    //***// This is the loop over time, should stay always the same
    // for reservoir i_step = (0 .. 249) * 10,  n_step = 10 =>
    i_step += buffoffzet;
    for (unsigned int t = i_step; t < (i_step + n_step); t++)
    {
//        printf("t%d\n", t);

    //***// This is the loop over nodes, which also should stay the same
        for (int i_node = 0; i_node < n_node; i_node++)
        {
            c_pop0 = 0.0f;
            c_pop1 = 0.0f;

            if (t == (i_step)){
                tavg(i_node + 0 * n_node) = 0;
                tavg(i_node + 1 * n_node) = 0;
            }

             // node isolation is not necessary as info is encoded in membrane pot
            // which is not taken into account with global coupling due to cv = 0
           // #define state(time, i_node) (state_pwi[((time) * 2 * n_node + (i_node))*size + id])
            r = state((t) % nh, i_node + 0 * n_node) + .01;
            V = state((t) % nh, i_node + 1 * n_node);


//            if (i_node == 10) {
////                if (t >= onset && t < (onset + duration)) {
//                if (t == 2500) {
//                    printf("we zijn er ewl");
//                    V += 10;
//                }
//            }
                int GO = 0;
            if (t>0 && t < inject_dyn) {

                // set to zero for significance testing. without input in training
                // but with boot up lorentz in prediciton phase
                int dynadd = train_or_pred;
                if (dynadd == 1) {
                // driving phase for 500 ts, encoding the info in membrane potential
                    if (i_node == 10) {
                            r = (weight * x[t]) + offsetab;
//                            V = (weight * x[t] + offsetab);
                    }
    //
                    if (i_node == 17) {
                            r =( weight * y[t]) + offsetab;
//                            V = weight * y[t] + offsetab;
                    }
    //
                    if (i_node == 30) {
                            r =( weight * z[t]) + offsetc;
//                            V = weight * z[t] + offsetc;
                    }
                }

//                else if (t >= inject_dyn || t == 0) {
//                    if (i_node == 10) {
//                            r = 4;
////                            V = -.5;
//                    }
//    //
//                    if (i_node == 17) {
//                            r = reststater;
////                            V = -.5;
//                    }
//    //
//                    if (i_node == 30) {
//                            r = reststater;
////                            V = -.5;
//                    }
//                }
                // input on the output nodes of shited in time signal
                // set to zero for significance testing. without input in training
                int teachadd = 0;
                if (teachadd == 1) {
//                if (t>10 && t < inject_dyn) {
                    if (i_node == 16) {
//                            r = (weight * x[t+teacherdelay]);// + offsetab;
                            r = (weight * xt[t]);// + offsetab;
//                            V = (weight * xt[t] + offsetab);
                    }
    //
                    if (i_node == 21) {
//                            r = (weight * y[t+teacherdelay]);// + offsetab;
                            r = (weight * yt[t]);// + offsetab;
//                            V = weight * yt[t] + offsetab;
                    }
    //
                    if (i_node == 23) {
//                            r = (weight * z[t+teacherdelay]);// + offsetc;
                            r = (weight * zt[t]);// + offsetc;
//                            V = weight * zt[t] + offsetc;
                    }
                }
            }
//              no preddelay
//                else if (t >= (inject_dyn+buffoffzet+160) && GO == 1) {
//                preddelay = 100
                else if (t >= (inject_dyn+buffoffzet+387) && GO == 1) {

                    // add or replace? replace with learned thoughts for now
//                    int thoughtidx = (t - buffoffzet) % thoughtsize;
                    int thoughtidx = ((t - buffoffzet) % thoughtsize + thoughtsize) % thoughtsize;
//                    thoughtidx = (thoughtidx - thoughtidx % thoughtsize) % thoughtsize;

                    if (i_node == 10) {
                        r = thougths[thoughtidx+thoughtsize*0];
//                                                    printf("thoughts %d\n", thoughtidx);


//                        if (id == 4 && t == (3300 + buffoffzet)){
                        if (id == 4){
//                           if (t == (inject_dyn+buffoffzet))
//                            {r = 30;
//                            printf("kom toch hier?\n");
//                            }
//                            printf("thoughts %d\n", thoughtidx);
//                            printf("t %d\n", t);
//                            printf("(inject_dyn+buffoffzet)+10 %d\n", (inject_dyn+buffoffzet)+10);


//                            printf("contentss %f\n", thougths[thoughtidx]);
//                            }
//                            r = (r + thougts[t+thoughtsize*0])/2;

//                            printf("rrr0% f\n", r);
//                            V = -.5;
                         }
                    }
    //
                    if (i_node == 17) {
//                            r = (r + thougts[t+thoughtsize*1])/2;
                            r = thougths[thoughtidx+thoughtsize*1];
//                            printf("rrr1%f\n", r);
//                            V = -.5;
                    }
    //
                    if (i_node == 30) {
//                            r = (r + thougts[t+thoughtsize*2])/2;
                            r = thougths[thoughtidx+thoughtsize*2];

                            if (id == 4 && t == (3300 + buffoffzet)){
//                                printf("rrr2 %f\n", r);
//                              V = -.5;
                            }
                    }
                }
//            }

            r = wrap_it_r(r);
            V = wrap_it_V(V);


            // This variable is used to traverse the weights and lengths matrix, which is really just a vector. It is just a displacement. /
            unsigned int i_n = i_node * n_node;

            for (unsigned int j_node = 0; j_node < n_node; j_node++)
            {
                //***// Get the weight of the coupling between node i and node j
                float wij = weights[i_n + j_node]; // nb. not coalesced
                if (wij == 0.0)
                    continue;

                // Get the delay between node i and node j
                dij = lengths[i_n + j_node] * rec_speed_dt;
                dij = dij + 0.5;
                dij_i = (int)dij;

                //***// Get the state of node j which is delayed by dij
                r_j = state(((t - dij_i + nh) % nh), j_node + 0 * n_node);
                V_j = state(((t - dij_i + nh) % nh), j_node + 1 * n_node);

//                if (i_node == 0 && id==0 && t == 2499)
//                    printf("indx %d\n", (t - dij_i + nh) % nh);
//                    printf("t %d\n", t);

                // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
                c_pop0 += wij * 1 * r_j;
                c_pop1 += wij * 1 * V_j;
            } // j_node */

            // global coupling handling, rec_n used to scale nodes
            c_pop0 *= global_coupling;
            c_pop1 *= global_coupling;

            // additive white noise generation in tvb (random_stream.normal(size=shape));
            // dont add noise to the input node
            int notsigi = 0;
            if ((i_node == 10 || i_node == 17 || i_node == 30 || i_node == 16 || i_node == 21 || i_node == 23) && notsigi) {
//                if (i_node == 10 || i_node == 17 || i_node == 30) {
                    noise_r = 0;
                    noise_V = 0;
//                }
            }
            else {
                sqrtdt = sqrt(dt);
                noise_r = sqrtdt * curand_normal(&crndst);
                noise_V = sqrtdt * curand_normal(&crndst);
                // gfun in tvb
                noise_r *= sqrt(2.0 * nsig);
                noise_V *= sqrt(2.0 * nsig * 2.0);

            }


             // DYNAMICS START //

            // scale the effect of the dynamics
//            float lambda = .1;

            //scales nice
            float lambdar = 1;
            float lambdaV = 1;

            if (heun_integrated == 1) {
                // Integrate with Heun (2nd order)
                dr = (1/tau * (Delta / (pi * tau) + 2 * V * r));
                dV = 1/tau * (powf(V, 2) - powf(pi, 2) * powf(tau, 2) * powf(r, 2) + eta + J * tau * r + I + cr * c_pop0 + cv * c_pop1);

                dr_i = r + dr * dt + noise_r;
                dV_i = V + dV * dt + noise_V;
    //            dr_i = r + dr * dt;
    //            dV_i = V + dV * dt;

                dr = dt/2.0 * (dr + (1/tau * (Delta / (pi * tau) + 2 * dV_i * dr_i)));
                dV = dt/2.0 * (dV + (1/tau * (powf(dV_i, 2) - powf(pi, 2) * powf(tau, 2) * powf(dr_i, 2) + eta + J * tau * dr_i + I + cr * c_pop0 + cv * c_pop1)));

                // No noise is added because it is not present in model
                r += dr + noise_r;
                V += dV + noise_V;
    //            r += dr;
    //            V += dV;
            }
            else if (euler_integrated == 1) {

                dr = (1/tau * (Delta / (pi * tau) + 2 * V * r));
//                dr = (1/tau * (Delta / (pi * tau) + 2 * V * r));
//                dr = V * r;
                dV = 1/tau * (powf(V, 2) - powf(pi, 2) * powf(tau, 2) * powf(r, 2) + eta + J * tau * r + I + cr * c_pop0 + cv * c_pop1);
//                dV = cr * c_pop0 + cv * c_pop1;

//                noise_r = 0;
//                noise_V = 0;
//                r += lambda * (dr * dt) + noise_r;
                r += lambdar * (dr * dt) + noise_r;
//                r += .001 * r; //+ noise_r;
                V += lambdaV * (dV * dt) + noise_V;
            }

            // DYNAMICS END //

            // Update the state
            state((t + 1) % nh, i_node + 0 * n_node) = r;
            state((t + 1) % nh, i_node + 1 * n_node) = V;

            // Update the observable
            tavg(i_node + 0 * n_node) += r/n_step;
            tavg(i_node + 1 * n_node) += V/n_step;

            // sync across warps executing nodes for single sim, before going on to next time step
//            __syncthreads();

        } // for i_node
    } // for t

// cleanup macros/*{{{*/
#undef params
#undef state
#undef tavg/*}}}*/

} // kernel integrate

// defaults from Stefan 2007, cf tvb/analyzers/fmri_balloon.py
#define TAU_S 0.65f
#define TAU_F 0.41f
#define TAU_O 0.98f
#define ALPHA 0.32f
#define TE 0.04f
#define V0 4.0f
#define E0 0.4f
#define EPSILON 0.5f
#define NU_0 40.3f
#define R_0 25.0f

#define RECIP_TAU_S (1.0f / TAU_S)
#define RECIP_TAU_F (1.0f / TAU_F)
#define RECIP_TAU_O (1.0f / TAU_O)
#define RECIP_ALPHA (1.0f / ALPHA)
#define RECIP_E0 (1.0f / E0)

// "derived parameters"
#define k1 (4.3f * NU_0 * E0 * TE)
#define k2 (EPSILON * R_0 * E0 * TE)
#define k3 (1.0f - EPSILON)

__global__ void bold_update(int n_node, float dt,
                      // bold.shape = (4, n_nodes, n_threads)
            float * __restrict__ bold_state,
                      // nrl.shape = (n_nodes, n_threads)
            float * __restrict__ neural_state,
                      // out.shape = (n_nodes, n_threads)
            float * __restrict__ out)
{
    const unsigned int it = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x;
    const unsigned int nt = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

    int var_stride = n_node * nt;
    for (int i_node=0; i_node < n_node; i_node++)
    {
        float *node_bold = bold_state + i_node * nt + it;

        float s = node_bold[0 * var_stride];
        float f = node_bold[1 * var_stride];
        float v = node_bold[2 * var_stride];
        float q = node_bold[3 * var_stride];

        float x = neural_state[i_node * nt + it];

        float ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0f);
        float df = s;
        float dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA));
        float dq = RECIP_TAU_O * (f * (1.0f - pow(1.0f - E0, 1.0f / f))
                * RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v));

        s += dt * ds;
        f += dt * df;
        v += dt * dv;
        q += dt * dq;

        node_bold[0 * var_stride] = s;
        node_bold[1 * var_stride] = f;
        node_bold[2 * var_stride] = v;
        node_bold[3 * var_stride] = q;

        out[i_node * nt + it] = V0 * (    k1 * (1.0f - q    )
                                        + k2 * (1.0f - q / v)
                                        + k3 * (1.0f -     v) );
    } // i_node
} // kernel
