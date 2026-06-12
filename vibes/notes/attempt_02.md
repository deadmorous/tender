# Tender core - attempt 2

Attempt 1 of Tender software is a result of work on vibes 1-25, stopped at
Phase 13.92 implemented. It turns out that the engine architecture is not strong
enough to address regular needs. Too many times when it was necessary
to make a derivation we ended up needing to implement a specific component
to fill a particular gap. That process did not seem to converge and
made code of the core engine quite messy.

I decided to start another attempt on the core. Achieving desired state
incrementally by updating the current state does not seem practical. Therefore,
we will start from scratch, but will keep all the existing code nearby,
borrowing from it whenever we need.

Let's reorganize code as follows.

- Create top-level directory named `attic/attempt_01`
- Move  `benchmark`, `src`, `examples`, `python`, `tests` to `attic/attempt_01`
- Move `vibes/0*` to `attic/attempt_01/vibes`
- Create `src`, copy `rational.hpp` to `src/include/tender`
- Create `tests`, copy unit tests for Rational and Polynomial there
- Setup CMake files: remove (from CMakeLists.txt) references to source files
  that we no longer have
- What we have now must be buildable if TENDER_BUILD_ATTEMPT_01 is set to true
  and have a prefix that lets identify targets and binaries as belonging
  to attempt 1. There must be zero interdependencies across attempts,
  just the root CMakeLists.txt should include our attempts.

By following the procedure I propose we enforce the principle to keep system
alive at every step, as if it were a plant. We just put what we have grown
aside and will start growing another plant, because it is simpler.
We will learn lessons coming from attempt 1. I will think on new core design
and will implement it either myself or with your help. We will revisit
all the goals discussed in vibes 1-25.

A note on further vibes numbering: use contiguous numbering, so attempt 2 will
have vibe 26 as its first one.