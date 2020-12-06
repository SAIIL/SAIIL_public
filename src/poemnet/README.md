# Table of Contents
* [POEMNet](#poemnet)
  * [Requirements](#requirements)
    * [Python](#python)
    * [R](#r)
    * [Other](#other)
* [Video statistics](#video-statistics)
  * [Overall videos' duration statistics](#overall-videos-duration-statistics)
    * [Example](#example)
  * [Phase duration statistics](#phase-duration-statistics)
    * [phase\_duration\.csv structure](#phase_durationcsv-structure)
    * [Make phase\_duration\.csv from anvil annotations](#make-phase_durationcsv-from-anvil-annotations)
    * [Make a phase\_factors\.csv](#make-a-phase_factorscsv)
    * [Generate statistics and plot](#generate-statistics-and-plot)
* [Inter\-annotator stats](#inter-annotator-stats)
  * [multiple\_annotator\.csv structure](#multiple_annotatorcsv-structure)
  * [Make multiple\_annotator\.csv from anvil annotations](#make-multiple_annotatorcsv-from-anvil-annotations)
  * [Generate inter\-annotator stats](#generate-inter-annotator-stats)
* [Model Results](#model-results)
  * [Per\-test\-video tsv file structure](#per-test-video-tsv-file-structure)
  * [Generate model result tsvs from saiil pkl file](#generate-model-result-tsvs-from-saiil-pkl-file)
    * [Saiil model output structure](#saiil-model-output-structure)
    * [Create directory of model results in tsvs](#create-directory-of-model-results-in-tsvs)
  * [Create surgical fingerprints](#create-surgical-fingerprints)
    * [Create fingerprint for each video analyzed](#create-fingerprint-for-each-video-analyzed)
    * [Create a side\-by\-side fingerprint](#create-a-side-by-side-fingerprint)
  * [Generate per\-phase performance metrics and confusion matrices](#generate-per-phase-performance-metrics-and-confusion-matrices)
    * [model\_metrics\.R generated files](#model_metricsr-generated-files)
  * [Per\-duration statistics](#per-duration-statistics)
* [Questions, comments, concerns, need help?](#questions-comments-concerns-need-help)
* [Citation](#citation)

# POEMNet
Repository holds software used in
[Automated operative phase identification in peroral endoscopic myotomy](https://doi.org/10.1007/s00464-020-07833-9).
In particular, it houses the scripts used to generate the statistics and
visualizations used in the Results section of the paper. Documentation
for the scripts includes heavy commenting within each script, an
informative commit message per script, and the below README. If you
still have any questions, please contact me.

## Requirements

### Python
Python >= 3.6 required. The following extra packages will be needed for
some of the scripts. Install through your distribution's package
manager/PyPI/etc.

1. `docopt` (verified to work with version `0.6.2_5`)
2. `numpy` (verified to work with version `1.19.1`)

### R
All calculations for the paper were performed with R 3.6. The
following packages will also be required (recommend adding with
`install.packages()`)

1. `tidyverse` (verified to work with version `1.3.0`)
2. `irr` (verified to work with version `0.84.1`)
3. `caret` (verified to work with version `6.0-86`)
4. `docopt` (verified to work with version `0.6.1`)
5. `e1071` (missed dependency in the `caret` package)

### Other
The `video_lengths.py` script will require ``ffprobe`` which is
typically packaged with ``ffmpeg``.

# Video statistics
The following sections address how to calculate the results shared in
the Results subsection *Video information* of the paper.

## Overall videos' duration statistics
To calculate the mean, min, max, median, and pstdev on the overall
lengths of a directory full of videos, use `video_lengths.py`. Results
are output to `stdout` in unit of seconds, rounded to two decimal
places. Of note, this will require
[`ffprobe`](https://ffmpeg.org/ffprobe.html).

### Example
```
video_lengths.py /data/directory_holding_videos
``` 
will output:
```
For the videos in '/data/directory_holding_videos', in seconds:
The mean is: 1467.5
The pstdev is: 645.5
The min is: 822
The median is: 1467.5
The max is: 2113
```

## Phase duration statistics
Phase duration statistics are calculated from annotated ground truth,
with one annotation per video. Our annotation file format, due to
historical reasons, are all converted to the
[Anvil](http://www.anvil-software.org/) file format, which is in xml.

All statistics, though, are calculated from a csv file (detailed in
section below). Therefore to generate statistics and the boxplot of
phase duration, if your annotations are in anvil format, I provide
scripts to convert that information into a csv of the correct structure.
Otherwise if you stored your data in another format, I recommend
skipping the headache of your annotation format -> Anvil -> CSV and
instead write a program to generate a CSV directly from your annotation
format. If you need help with this, please contact me. Happy to
offer guidance and/or write a script.

### `phase_duration.csv` structure
The below table shows the general structure of the CSV that contains
annotated ground truth for an entire set of videos:

| **variable** | **class** | **description** |
|:--|:--|:------------|
| video | character | Filename of video annotated |
| frame | double | Frame number for the video. Starts at 0. Equivalent to seconds for this case | 
| annotator | character | Annotation for each frame (eg clip, cut, or NA if not annotated) done by the annotator generically named "annotator." Any column variable name is fine.|

### Make `phase_duration.csv` from anvil annotations
1. Into a directory (eg `annotations_dir`), move a single Anvil format
   annotation per video in the dataset
    - It is ok if the "coder" field in the anvil annotation has
      different annotator names, the next step will standardize these
2. Rename all "coder" fields in the anvil annotation files to "annotator"

    ```
    make_anvil_same_annotator.py annotations_dir standardized_name_dir
    ```
3. Generate `phase_duration.csv`

    ```
    anvil_to_frame_csv.py 1 standardized_name_dir phase_duration.csv
    ```
    - the `1` tells the script to generate only one row (aka frame) per
      second.

### Make a `phase_factors.csv`
Phase names in annotation files are typically abbreviated (eg
`muc_incis`) rather than there full human-readable name (eg
"Mucostomy"). Analysis programs also tend to order them alphabetically
in the display (eg "Mucostomy" will occur before "Submucosal
Injection"), which does not make sense as we would prefer the phases to
be temporally ordered. To fix both these issues, the statistics script
is fed a simple csv (`phase_factors.csv`) that specifies a short to full
name translation. In addition, the rows' order matters, so the first
row will be displayed first, and the last row will be displayed last in
any future graphs. You can use this to order your phases temporally for
example or any other arbitrary sorting.

The `phase_factors.csv` has the following structure:

| **variable** | **class** | **description** |
|:--|:--|:------------|
| `name` | character | Variable name used for the annotated phase in the annotation file, eg `mucos_close`|
| `full_name`| character | Full human readable name of the phase, eg "Mucosotomy Closure"|

An example `phase_factors.csv` for POEM is present in the `examples/`
directory.

### Generate statistics and plot
The script `overall_phase_stats.R` is used. It will generate a TSV with
the following summary statistics for each phase:

1. mean
2. sd (standard deviation)
3. min
4. Q1 (1st quartile)
5. med (median)
6. Q3 (3rd quartile)
7. max 
8. mad (median absolute deviation)

It can optionally generate a boxplot. With the command, you can
optionally invoke the time scale to be logarithmic. Phases will
be displayed on the Y-axis in the order they are specified in the
`phase_factors.csv`. Images output format (eg png, jpg, pdf) is
automatically inferred from the specified file extension given. In
general, `pdf` will provide optimal image quality (text will look the
best upon import into LaTeX). You can also specify the height and width
of the output boxplot. Below is an example invocation that outputs a TSV
with summary statistics in `durations.tsv` and a boxplot `boxplot.png`
that is a 12×8 cm image with a log scale for the time axis:

```
overall_phase_stats.R -f phase_factors.csv -o durations.tsv \
    -p -l 12 8 boxplot.png phase_duration.csv 
```

Invoking `overall_phase_stats.R -h` will print to stdout a help
message explaining all the command line options.

# Inter-annotator stats
The following sections address how to calculate the results in the
Results subsection *Inter-annotator reliability and agreement*.

As with the **Phase duration statistics** above, all statistics are
calculated from a csv file that holds all the annotations (detailed in
section below). Therefore to generate inter-annotator statistics, you
can either go from Anvil annotations to generate the csv (instructions
below) or generate it from your own annotations. If you need help with
that, please let me know!

## `multiple_annotator.csv` structure
The below table shows the general structure of the CSV that contains
annotated ground truth for an entire set of videos by multiple
annotators. Note, it is nearly identical to the csv `phase_duration.csv`
except that it can hold an infinite number of annotator name columns:

| **variable** | **class** | **description** |
|:---|:--|:------------|
| `video` | character | Filename of video annotated |
| `frame` | double | Frame number for the video. Starts at 0. Equivalent to seconds for this case | 
| `annotator_1's name` | character | Annotation for each frame (eg clip, cut, or NA if not annotated) done by the first annotator.|
| `annotator_2's name` | character | Annotation for each frame (eg clip, cut, or NA if not annotated) done by the second annotator.|
| `annotator_N's name` | character | Annotation for each frame (eg clip, cut, or NA if not annotated) done by the nth annotator.|

## Make `multiple_annotator.csv` from anvil annotations
1. Create a directory and move the annotations for a set of videos by
   multiple annotators into the directory
    - We have included the annotations performed by `DAH`, `ORM`, and
      `tmw` in the `examples/multiple_annotator_annotations` directory
    - Note that you will need each annotator's name in the anvil xml
      file to be the same between files (eg do not have TMW in one and
      TW in the other, this will count as different annotators)
    - Also, each video must be annotated by **all annotators** otherwise
      the csv generation script may not output correctly
2. Generate the `multiple_annotator.csv` from the annotations stored in
   the directory `multiple_annotator_annotations`

    ```
    anvil_to_frame_csv.py 1 multiple_annotator_annotations \
	    multiple_annotators.csv
    ```

## Generate inter-annotator stats
The script `interannotator_stats.R` is used to generate Krippendorff's
alpha coefficient (to calculate inter-annotator reliability over the
entire video) and Fleiss' kappa (to calculate inter-annotator agreement
on a per-phase basis). It uses the annotation data extracted into
`multiple_annotator.csv`. It also requires the `phase_factors.csv`
generated in previous portions of the readme in order to translate
phase names and order then by user preference. Each second is treated
as a different "diagnosis" by an annotator and compared then between
annotators. An example invocation that calculates the statistics and
outputs Fleiss' kappa into fleiss.csv and Krippendorff's alpha into
kripp.csv is below:

```
interannotator_stats.R -f fleiss.csv -k kripp.csv phase_factors.csv \
    multiple_annotators.csv 
```

Of note, to keep things consistent across calculation of Krippendorff's
alpha and Fleiss' kappa, we now count "Idle" aka "NA" time the same
across the two groups. Before in the Fleiss' kappa these times were not
included for the calculations. You will notice slightly different
results compared to our paper's table (in particular, Overall Fleiss'
kappa is the same as the Krippendorf's alpha, as anticipated).

# Model Results
The following section will show you how to perform the following:

1. Generate individual and side-by-side surgical fingerprint plots
   for each video in the test set (eg *Fig 3* in the paper)
2. Generate performance across phases statistics (per video in test set
   and all videos combined) (eg *Table 3* in the paper)
3. Generate a confusion matrix per video and across all videos
   in the test set (eg *Fig 4* in the paper)
4. Generate performance across phase-duration statistics on
   user-specified duration intervals (eg *Table 4* in the paper)

As in previous sections, calculations are done from a csv/tsv file, in
this instance, it's one tsv file per video in the test set for the
model. I will outline the structure of that tsv file and also show how,
from our typical model's output, you can generate the tsv file. Then I
will show how to generate the figures and statistics from the tsv files.

## Per-test-video tsv file structure
To generate summary statistics, surgical fingerprint plots, and
confusion matrices, there should be a directory full of tsvs. Each tsv
contains a row per video second, with each row containing information
below:

| **variable** | **class** | **description** |
|:---|:--|:------------|
| `second` | integer | second of the video |
| `vid_num` | integer | video's number in the test set | 
| `block` | integer | phase's block number. each time a phase transitions to another phase this increments| 
| `gt` | character | ground truth label for the phase | 
| `predicted` | character | predicted (most likely) phase| 
| `phase_1 name` | double | model's probability estimate that the current video second is `phase_1 name`| 
| `phase_2 name` | double | model's probability estimate that the current video second is `phase_2 name`| 
| `phase_... name` | double | model's probability estimate that the current video second is `phase_... name`| 
| `phase_N name` | double | model's probability estimate that the current video second is `phase_N name`| 

Replace the column variable names `phase_1 name`, `phase_2 name`, etc with
whatever the phase labels are that your model is trying to identify.

## Generate model result tsvs from saiil pkl file
This section describes how to generate tsvs detailed in the prior
section. In particular, it shows how to generate it from a `pkl` file
that our model outputs.

### Saiil model output structure
POEMNet's results on the test set are published in the
`examples/poemnet.pkl`. The pkl file contains a single python
dictionary with keys `class_names`, `gt`, `lstm:`, and `lstm_hmm`. These
contain:

- `class_names`: list of class (phase) names, eg `mucos_close`
- `gt`: a dictionary with keys `[1, 2, ..., 20]`, one for each video.
   Each key's value is a list of the ground truth labels, with
   the list's index corresponding to the video second.
- `lstm:`: a dictionary with keys `[1, 2, ..., 20]`, one for each
   video. Each key's value is a `numpy.ndarray`, with each index holding
   the likelihoods of that video's second being categorized as each
   of the different class names (aka phases).
- `lstm_hmm`: same as `lstm:` except the `lstm:` results with additional
  HMM smoothing added

To quickly ascertain the contents of the pkl, you can run the script
`pkl_dump.py` such as below:

```
pkl_dump.py poemnet_results.pkl
```
that will output:
```
Class names found in 'class_names' are:
	muc_incis
	mucos_close
	myotomy
	submuc_inject
	tunnel
A model's results for 20 videos found in: 'lstm_hmm'
Ground truth for 20 videos found in: 'gt'
A model's results for 20 videos found in: 'lstm:'
```
This information will come in handy when you run
`make_model_results_tsv.py` later.

### Create directory of model results in tsvs
Now that we know the pkl structure, we can run
`make_model_results_tsv.py` to generate a directory full of TSVs for the
model's results (without HMM smoothing) called `video_tsvs` for further
analysis:

```
make_model_result_tsvs.py -c 'class_names' -p poemnet_results.pkl \
    -g 'gt' -m 'lstm:' video_tsvs
```
The command-line options are documented by invoking the script with
`-h`. 

## Create surgical fingerprints
Surgical fingerprints are a way to display the model's phase likelihoods for
each second of the video compared to the annotated ground truth. An
example is in *Fig 3* of the paper. Below we will show you how to
generate a fingerprint for each video the model analysed, and how to
create a "side-by-side" fingerprint like in *Fig 3*.

Both use the same script, `fingerprints.R`. They will require a
`phase_factors.csv` file already written as documented above and a
directory full of model result TSVs (one per video).

### Create fingerprint for each video analyzed
Creating a fingerprint per each video allows you to rapidly analyze the
model's performance on a per-video basis and try to target areas that
will need improvement. To create one per video, do the following:

1. Make a directory to contain the scripts output

    ```
    mkdir results
    ```
2. Run `fingerprints.R all` to generate a fingerprint per-video, into
   `results/`, eg:

    ```
    fingerprints.R all -o results/ -f phase_factors.csv -t video_tsvs/
    ```
    Optionally you can also specify image width, height, and format (use
    the `-h` command line switch to see fully how to use the script).

### Create a side-by-side fingerprint
As in the POEMNet paper, you may want to generate a side-by-side
comparison of two fingerprints to highlight differences in model
analysis of each case. To do so (such as we did to generate the figure
in the paper), do the following:

```
fingerprints.R two -W 15 -H 7.5 -o results/ \
    -f phase_factors.csv -t video_tsvs \
    video_08.tsv "Straightforward" video_10.tsv "Tortuous Esophagus"
```
This will generate a 15×7.5 image called
`results/video_08_video_10_fingerprint.png` that shows the fingerprint
for video 08 on the left with the title "Straightforward" and the
fingerprint for video 10 on the right with the title "Tortuous
Esophagus".

## Generate per-phase performance metrics and confusion matrices
To generate per-phase precision, recall, F1 score, and prevalence
statistics (*Table 3* in the paper) and recall/precision confusion
matrices (seen in POEMNet's *Fig 4*), follow the below steps:

1. Create a directory of TSVs, with one per model result on a test
   set video, as detailed above. For example, create a `video_tsvs`
   directory.
2. Make a directory to store the per-phase and confusion matrix results:

    ```
    mkdir results_dir
    ```
3. Generate model metrics and confusion matrices:

    ```
    model_metrics.R -c -o results_dir/ -f phase_factors.csv video_tsvs/
    ```
    Of note, in the paper we used a width of 12 and height of 8 which
    you can specify as a command option (see output from `-h` for help)

### `model_metrics.R` generated files
`model_metrics.R` generates a number of files that analyze the model's
performance on the test set. Assuming you followed the above, the
`results_dir/` will contain, for each video and for all videos combined,
the following files (labeled either `combine_foo` or `video_NN_foo`), 
where `foo` can be:

1. `raw.tsv`: Raw results from analysis with `caret` passed through
   `broom::tidy()`. This includes overall accuracy (with CI) and
   per-phase recall, precision, etc.
2. `perclass.tsv`: A tsv that shows, per-phase and overall, the
   model's precision, recall, f1-score, and prevalence. The one for
   combined is identical to *Table 3* in the paper
3. `confusion.tsv`: A tsv that puts the caret results into a "tidy"
   format from which to generate the plots. It includes both the number
   and proportion of phases classified by the model as each phase.
4. `precision.png`: If specified to generate, an image of the precision
   confusion matrix.
5. `recall.png`: If specified to generate, an image of the recall
   confusion matrix. The one for combined is identical to the confusion
   matrix in *Fig 4* of the paper.

## Per-duration statistics
We noticed for the shorter phases that our model had lower performance.
To clearly convey this, we created a script to generate statistics on
accuracy called `per_block_duration_accuracy.R`. What this does is
examine the model's accuracy across blocks of different lengths. A block
is a continuous phase in the surgery that is bounded by different
phases. For example, if a surgeon "injects the submucosa" then "does a
mucosotomy," notices that they need more injection, so "injects the
submucosa" again, this counts as three different blocks. This allows us
to analyze short continuous segments, even if they typically are a long
phase.

To analyze different block lengths (eg from 1-30 seconds, 31-60 seconds,
etc), you need to create a csv file that holds your preferred block
lengths that has the following structure:

| **variable** | **class** | **description** |
|:-|:-|:------------|
| `start` | double | Start time, in seconds, for the current interval|
| `end` | double | End time, in seconds, for the current interval. To specify until the end of the video, use 'Inf' which stands for infinity in R|

An example `durations.csv` exists in `examples/`.

To generate per-duration model accuracy statistics (as seen in *Table 4*
of the Results section of the POEMNet paper), do the following:

1. Create a directory of TSVs, with one per model result on a test
   set video, as detailed previously. For example, create a `video_tsvs`
   directory.
2. Make a `durations.csv` as above
3. Run `per_block_duration_accuracy.R`

    ```
    per_block_duration_accuracy.R -i durations.csv \
	    -o per_block_duration_accuracy.tsv video_tsvs
    ```

The output file `per_block_duration_accuracy.tsv` will contain:
```
start  end  accuracy
1      30   0.4178082191780822
31     60   0.6431818181818182
61     300  0.7601941747572816
301    600  0.9359582542694497
600    Inf  0.8777989802916009
```
Which is identical to *Table 4* results for POEMNet in the paper! If you
also want to generate the results for after HMM post-processing, you
will need to generate another TSV directory for the `lstm_hmm` model and
then run `per_block_duration_accuracy.R` on this directory as well.

# Questions, comments, concerns, need help?
Please contact me in the communication medium of your preference listed on my
[Contact page](https://thomasward.com/contact/).

# Citation
If you found the code helpful for your research, please cite our paper:
```
@article{wardAutomatedOperativePhase2020,
  title = {Automated Operative Phase Identification in Peroral Endoscopic Myotomy},
  author = {Ward, Thomas M. and Hashimoto, Daniel A. and Ban, Yutong and Rattner, David W. and Inoue, Haruhiro and Lillemoe, Keith D. and Rus, Daniela L. and Rosman, Guy and Meireles, Ozanan R.},
  year = {2020},
  month = jul,
  issn = {1432-2218},
  doi = {10.1007/s00464-020-07833-9},
  journal = {Surgical Endoscopy},
  language = {en}
}
```
