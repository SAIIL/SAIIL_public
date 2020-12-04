#!/usr/bin/env Rscript
"Calculates Fleiss' Kappa and Krippendorff's alpha for a set of videos
annotated by multiple annotators to determine the inter-annotator
statistics. Treats each frame/second of the annotation as a diagnosis of
a certain phase, with not annotated frames (aka 'Idle' ones) holding
the diagnosis of 'Idle.' Fleiss' kappa (overall) and Krippendorff's alpha
will tend to have the same value if the videos are long enough, but the Fleiss'
kappa will return a per-phase agreement.

Usage: interannotator_stats.R [-f FLEISSRESU] [-k KRIPPRESU] FACTORS GTS

Options:
	-f FLEISSRESU  Filename to save CSV of Fleiss's kappa results.
	-k KRIPPRESU  Filename to save CSV of Krippendorf.

Arguments:
	FACTORS  CSV file that holds short to long name map and ordered in user preferences that save files will keep (eg temporally).
	GTS  CSV file that holds GT for multiple annotators.
" -> doc

library(docopt)
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(irr))

calculate_fleiss <- function(df, factor_file) {
  # get levels to order the output. Need to add an "Overall" level to the
  # phases since we also want to report this
  phase_levels <- c(get_levels(factor_file), "Overall")

  # kappam.fleiss outputs list with $detail being per-phase numbers and
  # $value being overall kappa
  results <- kappam.fleiss(df, detail = TRUE)

  # build tibble with all the phases and overall kappa values
  results$detail %>%
    as_tibble(.name_repair = "universal") %>%
    transmute(phase = ...1, stat = ...2, kappa = n) %>%
    # toss out z and p values that are useless
    filter(stat == "Kappa") %>%
    select(phase, kappa) %>%
    add_row(phase = "Overall", kappa = results$value) %>%
    mutate(kappa = round(kappa, digits = 3)) %>%
    mutate(phase = parse_factor(phase, levels = phase_levels)) %>%
    translate_phases(factor_file) %>%
    # sort rows in order of phases (order given by user in provided factor csv)
    arrange(phase)
}

calculate_kripp <- function(df) {
  # kripp alpha expects raters as the rows and observations as the columns
  # so transpose the dataframes
  # kripp.alpha spits out an erroneas NA coercian warning (there are no NAs in the
  # dataset so suppress it
  kripp <- suppressWarnings(kripp.alpha(t(df), method = c("nominal")))

  # stores value in value, then make that into a tibble for easier writing and only
  # keep the 2nd column which is the column that will store the value
  kripp$value %>%
    enframe() %>%
    transmute(Krippendorf_alpha = signif(value, digits = 3))
}

translate_phases <- function(df, factor_file) {
  # csv structure nicely allows us to create a map (named vector in R) with deframe()
  # fct_recode will later want this in full_name, name order so flip the columns around
  # with select
  phase_translation <- read_csv(factor_file, col_types = "cc") %>%
    select(full_name, name) %>%
    # add a phase idle in the name mapping
    add_row(full_name = "Idle", name = "Idle") %>%
    # add overall phase in the name mapping
    add_row(full_name = "Overall", name = "Overall") %>%
    deframe()

  df %>%
    # translate short_names to long pretty names
    mutate(phase = fct_recode(phase, !!!phase_translation))
}



get_levels <- function(factor_file) {
  # Factor info has the correct order of the phases already (user entered
  # them in the csv file in the correct order. Add a level "Idle" which will
  # replace NA in the non-annotated seconds
  read_csv(factor_file, col_types = "cc") %>%
    .[["name"]] %>%
    c(., "Idle")
}

make_gt <- function(gt_file, factor_file) {
  # factor_file is a csv with structure:
  # 	header: name, full_name
  # Phases are listed in temporal order in the csv.
  phase_levels <- get_levels(factor_file)


  # csv structure is video, frame, annotator 1's annotations, ...., annotation N's annotations
  # by specifying a default, I allow it to translate any number
  read_csv(gt_file, col_types = cols(
    video = col_character(),
    frame = col_integer(),
    .default = col_character()
  )) %>%
    # select only annotator annotated GTs, remove video and frame columns
    select(-video, -frame) %>%
    # get rid of seconds where everyone did not annotate (treat these padding times, like the
    # start, end, and down time parts of the case, essentially as an idle, so we then only look
    # at when at least one person annotated something. Not a perfect measure but gets rid of some
    # "automatic" agreement we get when nothing is definitively happening in a video
    # CAVEAT I stopped doing this; methodologically, the annotator had to watch that anyways and
    # make a decision (albeit an easy one) so I ignored it. I improved the trimming of videos
    # substantially to not include excess at the beginning and end so this should improve
    # filter_all(any_vars(!is.na(.))) %>%
    # change NA to Idle
    mutate_all(replace_na, "Idle") %>%
    # translate phases into correct order by making them factors, keep NA's to represent "idle"
    mutate_all(parse_factor, levels = phase_levels)
}

main <- function(gt_file, factor_file, fleiss_file, kripp_file) {
  gts <- make_gt(gt_file, factor_file)

  if (!is_null(fleiss_file)) {
    gts %>%
      calculate_fleiss(factor_file) %>%
      write_csv(fleiss_file, col_names = TRUE)
  }
  if (!is_null(kripp_file)) {
    gts %>%
      calculate_kripp() %>%
      write_csv(kripp_file, col_names = TRUE)
  }
}

opt <- docopt(doc)
main(opt$GTS, opt$FACTORS, opt$f, opt$k)
