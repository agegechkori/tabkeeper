use std::io::IsTerminal;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// A progress bar when writing to a terminal. Otherwise, e.g. when output
/// goes to a file, only the per-page lines are printed.
pub struct Progress {
    bar: Option<ProgressBar>,
    /// Lines go to stderr, keeping stdout for the JSON report (`--json`).
    to_stderr: bool,
}

impl Progress {
    pub fn new(total: usize, to_stderr: bool) -> Self {
        // The bar itself is drawn on stderr.
        let terminal = if to_stderr {
            std::io::stderr().is_terminal()
        } else {
            std::io::stdout().is_terminal()
        };
        if !terminal {
            return Self { bar: None, to_stderr };
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
        Self {
            bar: Some(bar),
            to_stderr,
        }
    }

    /// Prints a line above the bar.
    pub fn line(&self, text: &str) {
        match &self.bar {
            Some(bar) => bar.println(text),
            None if self.to_stderr => eprintln!("{text}"),
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
