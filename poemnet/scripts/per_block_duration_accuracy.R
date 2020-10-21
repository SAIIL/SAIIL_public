#!/usr/bin/env Rscript
"Takes a CSV of durations, with columns start and end, then calculates, on a per-block duration,
the mean accuracy for those blocks. It then outputs a CSV with the same input csv with an additional
column added for accuracy.

Usage: per_block_duration_accuracy.R [-h] [-o OUTFILE] -i DURATIONS TSVDIR

Options:
	-h             Print this menu and exit.
	-i DURATIONS   File, in CSV format, that contains block duration ranges to analyse accuracy for.
	-o OUTFILE     Filename to put results in TSV format [default: per_block_duration_accuracy.tsv].

Arguments:
	TSVDIR         Directory full of TSVs on model results per video. TSVS need, at minimum, columns (vid_name, gt, predicted)
" -> doc

library(docopt)
suppressPackageStartupMessages(library(tidyverse))

# takes a dataframe and a vector with lower and upper limit of
# block duration length (in rows, aka seconds since one row per second)
block_duration_range_accuracy <- function(limits, df) {
  df %>%
    # since blocks are 1,...N per video, group by vid_num *and* block
    # this calculates length of the block (length in number of rows)
    add_count(vid_num, block, name = "length") %>%
    # only look when length is between specified limits
    filter(between(length, limits$start, limits$end)) %>%
    # create boolean column when model predicted correctly
    mutate(correct = gt == predicted) %>%
    # mean of a boolean column is same as % correct
    summarise(accuracy = mean(correct)) %>%
    # only want to return a dbl, not a tibble so pull()
    pull(accuracy)
}

get_durations <- function(durations_file) {
  read_csv(durations_file, col_names = TRUE, col_types = "dd") %>%
    # in order to have a list of ((start_01, end_01), (start_02, end_02)... (start_nn, end_nn))
    # which we can pass to block_duration_range_accuracy(), need to transpose the tibble
    # as a side-effect, this gives a nice named vector with names c("start", "end")
    transpose()
}


get_video_results <- function(tsv_dir) {
  # all videos in format video_NN.tsv with header:
  # (second, vid_num, block, gt, predicted, step_01, ..., step_NN)
  dir(path = tsv_dir, pattern = "^video_\\d{2}.tsv$", full.names = TRUE) %>%
    map(read_tsv, col_types = cols(
      second = col_integer(),
      vid_num = col_integer(),
      block = col_integer(),
      gt = col_character(),
      predicted = col_character(),
      .default = col_double()
    )) %>%
    bind_rows() %>%
    select(vid_num, block, gt, predicted)
}

# takes the durations in_file and adds a column for the accuracies
# and saves it
create_outfile <- function(accuracies, durations_file, out_file) {
  read_csv(durations_file, col_names = TRUE, col_types = "dd") %>%
    # enframe will take accuracies, an atomic vector of doubles, and
    # make it into a single column with title "accuracy"
    bind_cols(enframe(accuracies, name = NULL, value = "accuracy")) %>%
    write_tsv(path = out_file, col_names = TRUE)
}

main <- function(opts) {
  get_durations(opts$i) %>%
    map_dbl(block_duration_range_accuracy, get_video_results(opts$TSVDIR)) %>%
    create_outfile(opts$i, opts$o)
}

opt <- docopt(doc)
main(opt)
