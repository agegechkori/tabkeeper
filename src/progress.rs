use std::io::IsTerminal;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// A progress bar when writing to a terminal. Otherwise, e.g. when output
/// goes to a file, only the per-page lines are printed.
pub struct Progress {
    bar: Option<ProgressBar>,
}

impl Progress {
    pub fn new(total: usize) -> Self {
        if !std::io::stdout().is_terminal() {
            return Self { bar: None };
        }
        let bar = ProgressBar::new(total as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner} {elapsed_precise} [{wide_bar}] {pos}/{len} · {msg} · {eta} left",
            )
            .expect("valid template")
            .progress_chars("=> "),
        );
        bar.enable_steady_tick(Duration::from_millis(250));
        Self { bar: Some(bar) }
    }

    /// Prints a line above the bar.
    pub fn line(&self, text: &str) {
        match &self.bar {
            Some(bar) => bar.println(text),
            None => println!("{text}"),
        }
    }

    pub fn advance(&self, message: String) {
        if let Some(bar) = &self.bar {
            bar.inc(1);
            bar.set_message(message);
        }
    }

    pub fn finish(&self) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }
}
