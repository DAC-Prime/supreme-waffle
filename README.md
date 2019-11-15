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

## GCP Configuration + Docker Setup
1. Open [GCP Setup Note](https://colab.research.google.com/drive/1L5rXPmC-DwbRVXGZkF5pZNNGQfGP6YhB)
2. Follow instructions to create project and VM instance. You can also check [Screenshots](https://drive.google.com/open?id=1cpV68nUwkHCCgmH70DTOMXMrzUqCqins) for configuration .
3. After SSH, download [DAC repo](https://github.com/DAC-Prime/supreme-waffle) into your home directory
7. Run `docker build -t dac .` to build a image. This should take a few minutes.
8. Run `docker run -it dac /bin/bash` to get a bash session for this image.
9. Remember to STOP your instance after use. Don't waster your MONEY!
