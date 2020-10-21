#!/usr/bin/env Rscript
"Calculates duration mean and standard deviation per phase and optionally
outputs a boxplot. Requires 'tidyverse' and 'docopt' packages.

Usage:
	overall_phase_stats.R -f FACTORFILE -o OUTFILE [(-p [-l] WIDTH HEIGHT PLOTFILE)] GT

Options:
	-f  File that contains short to long name translation and phase ordering
	-o  File to write tsv of per-phase durations' statistics
	-p  Flag to indicate you want a boxplot generated of width, height, format saved to output
	-l  Flag to indicate you want a boxplot plotted on a log-scale

Arguments:
	FACTORFILE	File containing short to longname translation. Phases are in temporal order.
	OUTFILE	File to write TSV of per-phase durations' statistics
	WIDTH	Width of boxplot in cm
	HEIGHT Height of boxplot in cm
	PLOTFILE	Filename for saved boxplot (image format inferred from file-extension)
	GT	CSV file containing per-second ground-truth of all videos
" -> doc

library(docopt)
suppressPackageStartupMessages(library(tidyverse))

generate_per_video_phase_durations <- function(df) {
  df %>%
    group_by(video, annotation) %>%
    summarise(duration = n())
}

generate_per_video_stats <- function(df) {
  df %>%
    group_by(annotation) %>%
    summarise(
      mean = round(mean(duration), digits = 2),
      sd = round(sd(duration), digits = 2),
      min = min(duration),
      Q1 = quantile(duration, probs = 0.25),
      med = median(duration),
      Q3 = quantile(duration, probs = 0.75),
      max = max(duration),
      mad = round(mad(duration), digits = 2)
    )
}

make_levels <- function(factor_info) {
  # Factor info has the correct order of the phases already (user entered
  # them in the csv file in the correct order. Since we want to display a boxplot with
  # the temporal "first" step on the top, just extract the 'name'
  # CAVEAT: will need to reverse it when it's a boxplot since "first" is actually a higher value on a y-axis)
  factor_info %>%
    .[["name"]]
}

make_gt <- function(gt_file, factor_file) {
  # factor_file is a csv with structure:
  # 	header: name, full_name
  # Phases are listed in temporal order in the csv.
  factor_info <- read_csv(factor_file, col_types = "cc")
  phase_levels <- make_levels(factor_info)
  # csv structure nicely allows us to create a map (named vector in R) with deframe()
  # fct_recode will later want this in full_name, name order so flip the columns around
  # with select
  phase_translation <- factor_info %>%
    select(full_name, name) %>%
    deframe()

  # gt file is a csv structured:
  # 	header: video,frame,annotator
  #   row1:uuid1.mp4, 0, phase_annotated
  # 	etc
  read_csv(gt_file, col_types = "cdc") %>%
    # get rid of NA seconds (aka non-annotated seconds)
    # cannot use these to generate stats unways because the script to make
    # the file does not know when the video ends (just writes annotations until
    # the last annotated second) so could miss unannotated seconds at the end
    filter(!is.na(annotator)) %>%
    # translate the phases into the correct order by making them factors
    mutate(annotation = parse_factor(annotator, levels = phase_levels)) %>%
    # translate short_names to long_names (use '!!!' to unpack the named vector)
    mutate(annotation = fct_recode(annotation, !!!phase_translation)) %>%
    # only keep what i want
    select(video, annotation)
}

plot_boxplot <- function(durations, wants_log_scale) {
  duration_plot <- durations %>%
    # flip order of steps so temporally first step is at top of boxplot
    mutate(annotation = fct_rev(annotation)) %>%
    ggplot(mapping = aes(x = annotation, y = duration)) +
    geom_boxplot() +
    coord_flip() +
    theme_classic() +
    labs(
      x = "Phase",
      y = "Duration (s)"
    ) +
    theme(axis.ticks.y = element_blank())

  if (wants_log_scale) {
    duration_plot + scale_y_log10()
  } else {
    duration_plot
  }
}

save_boxplot <- function(duration_plot, plot_width, plot_height, plot_file) {
  ggsave(plot_file, plot = duration_plot, width = plot_width, height = plot_height, units = c("cm"))
}

main <- function(gt_file, factor_file, out_file, wants_plot, wants_log_scale, plot_width, plot_height, plot_file) {
  durations <- make_gt(gt_file, factor_file) %>%
    generate_per_video_phase_durations()

  durations %>%
    generate_per_video_stats() %>%
    write_tsv(out_file, col_names = TRUE)

  if (wants_plot) {
    plot_boxplot(durations, wants_log_scale) %>%
      save_boxplot(plot_width, plot_height, plot_file)
  }
}

opt <- docopt(doc)
main(opt$GT, opt$FACTORFILE, opt$OUTFILE, opt[["-p"]], opt[["-l"]], as.numeric(opt$WIDTH), as.numeric(opt$HEIGHT), opt$PLOTFILE)
