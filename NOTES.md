look into uiPSF, i think what they do is similar to what is needed here

Setps for aberration correction:

- read image (optional)

- get masked images (optional)

- if masked images are of the same object (e.g. beads) then use as some sort of average

- Polynomial coefficient search:
    - iterate through values for coefficients
    - apply polynomial to masked image (bead)
    - compare to ground truth
    - repeat until best ground truth or some other stop criterion

- save parameters in some descriptive database


What is the ground truth?

The ground truth is some description of the shape of what is expected.

e.g. for a bead the tool description would be "circle with 1.0um diameter"
then teh tool will somehow measure how circular the aberration corrected image is and with what parameter and compare to desired

e.g. for some muscle fibre the description would be "cylinder with 0.2mm length and 1um diameter"
then the tool takes those measurements from the aberration corrected image and compares to the ground truth
