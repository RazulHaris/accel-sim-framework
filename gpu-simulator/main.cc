// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#include "accel-sim.h"

/* TO DO:
 * NOTE: the current version of trace-driven is functionally working fine,
 * but we still need to improve traces compression and simulation speed.
 * This includes:
 *
 * 1- Prefetch concurrent thread that prefetches traces from disk (to not be
 * limited by disk speed)
 *
 * 2- traces compression format a) cfg format and remove
 * thread/block Id from the head and b) using zlib library to save in binary
 * format
 *
 * 3- Efficient memory improvement (save string not objects - parse only 10 in
 * the buffer)
 *
 * 4- Seeking capability - thread scheduler (save tb index and warp
 * index info in the traces header)
 *
 * 5- Get rid off traces intermediate files -
 * change the tracer
 */

int main(int argc, const char **argv) {
  accel_sim_framework accel_sim(argc, argv);
  accel_sim.simulation_loop();

  printf("accel-sim: pre-756ed33 base+delta trace lines re-decoded = %llu\n",
         legacy_base_delta_lines());
  if (g_trace_paperera_decode)  // debug plan Stage 1.3
    printf("accel-sim: -trace_paperera_decode 1 (v1.2.0 decoding)\n");

  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}
