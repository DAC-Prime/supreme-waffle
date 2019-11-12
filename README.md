# supreme-waffle
Replicate of the DAC paper.

## Tasks
| Name         | Task    |
|--------------|---------|
| Yan Huang    | A2C     |
| Cancan Huang | AHP     |
| Zhe Hu       | DAC     |
| Zezhi Wang   | PPO     |
| Ziyao Huang  | GYM     |
| Yichen Chai  | IOPG\_OC |

## Set Up

Install Docker on your machine (or GCP instance).

Copy the mjkey.txt file from piazza into this folder.

Run `docker build -t dac .` to build a image. This should take a few minutes.

Run `docker run -it dac /bin/bash` to get a bash session for this image.

