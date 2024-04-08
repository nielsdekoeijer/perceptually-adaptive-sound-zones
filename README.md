# Block-Based Perceptually Adaptive Sound Zones with Reproduction Error Constraints
Contains the code corresponding to the publication ``Block-Based Perceptually Adaptive Sound Zones with Reproduction Error Constraints''. 

Unfortunately, certain parts have been left out deliberately for legal reasons (e.g. a MOSEK license and some of the audio files).

For my own sanity and reproduceability, we have used Docker to snap-shot the exact linux environment in which the experiment ran. It can be built by running `./scripts/build_docker.sh` and it is subsequently used in the `perform_*` scripts to run the experiments. You could also just install everything locally and just use the python scripts as is :-).
