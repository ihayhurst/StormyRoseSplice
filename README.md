# StormyRoseSplice
  - Mandelbrot plot play
  - The output array can be plotted in Matplotlib or Pillow, and saved to a .png
   - Matplotlib is nice as it's easy to add a colormap. I havent worked out quite how PIL does it yet

## Inspiration:
  - Prompted by Reddit post by Adam Reed and his [pastebin code](https://pastebin.com/mr8stzKx), which inspired a wave of nostalgia for early 90's Mandelbrot fever and my first ever pass at doing this in QBasic (After reading James Gleick's 'Chaos') Memories of the awesome fractint https://en.wikipedia.org/wiki/Fractint
  - Also, a chance to practice some code optimisations and try some of the mindblowing python that [JeanFrancoisPuget has done](https://www.ibm.com/developerworks/community/blogs/jfp/entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en)
  - This implements two methods from JeanFrancoisPuget's matrix. Generating the set by iterating through the coordinate space which is speeded up immensely using the jit decorator from numba, or by passing arrays to some OpenCL code to run on gpu

## Examples:
![Plot example 1](docs/plot1.png)
![Plot example 2](docs/plot2.png)
![Plot example 3](docs/plot3.png)
![Plot example 4](docs/plot4.png)
![Plot example 5](docs/plot5.png)
