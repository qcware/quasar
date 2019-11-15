import vulcan

import sys
m = int(sys.argv[1])
n = int(sys.argv[2])
# vulcan.run_timings_blas_1(m, n)
# vulcan.run_timings_gate_1(n)
# vulcan.run_timings_gate_2(n)
vulcan.run_timings_measurement(m, n)
