# Underwater fish detection tool
contributes: Dan and Mauro 

![](fish.gif)

## Install Python dependencies

```bash
pip3 install -r requirements.txt

```

## Manage tool

The manage tool is a utility script that runs some useful scripts. E.g;

```bash
python3 manage --help
python3 manage run_tests
```


## External Dependencies

To extract the motion vectors we need to compile `extract_mvs.c` from ffmpeg

```bash
make extract_mvs.c
```

And to use it
```
./extract_mvs ../data/example_video/20140626102131.m2ts > vectors.txt

```
