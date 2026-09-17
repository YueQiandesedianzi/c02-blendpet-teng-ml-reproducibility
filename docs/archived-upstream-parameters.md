# Archived upstream parameters

These settings describe archived code. They are not a verified exact match to every frozen source file.

The archived upstream algorithm reconstructs repeated or reset timestamps within contiguous valid blocks. A block with more than 1% duplicate intervals uses its span divided by n−1; otherwise it uses the median positive interval, with 0.001 s as fallback. It resamples at no more than 1,000 Hz. A third-order Butterworth bandpass (0.25 Hz to min(20 Hz, 0.45 fs)) supports periodicity and peak detection. The period search uses lags from 0.25 to 2.5 s. Peak spacing is at least max(0.20,0.60T) s and prominence is at least 0.45 times the robust filtered scale. Candidate boundaries lie midway between peaks. These settings document archived code, not a verified exact match to every frozen file.

Archived QC uses durations from max(0.20,0.55T) to min(2.8,1.60T) s; robust deviations of VPP, SNR proxy, and baseline no larger than four scales; and template correlation no smaller than max(0.20,the file 5th percentile). A file template and robust thresholds use candidate cycles from that file. The frozen per-cycle decisions, not a rerun with an unverified source version, define the pool in this revision. Source context may contain later evaluation cycles. File-local filtering is not independent-sample validation.

The numerical reproduction entry point is the frozen 256-point cycle plus its required metadata.
