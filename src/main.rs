use std::{error::Error};

use hound::WavReader;
use rustfft::{FftPlanner, num_complex::Complex};
use plotters::prelude::*;

const WINDOW_SIZE: usize = 1024;
const OVERLAP: f64 = 0.5;
const SKIP_SIZE: usize = (WINDOW_SIZE as f64 * OVERLAP) as usize;

fn main() -> Result<(), Box<dyn Error>> {
    let mut wav = WavReader::open("example.wav")?;
    let samples = wav
        .samples()
        .collect::<Result<Vec<i16>, _>>()?;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);

    let frames = samples.as_slice()
        .windows(WINDOW_SIZE) // You can use const generics here in the future!
        .enumerate().filter_map(|(i, f)| if i % SKIP_SIZE == 0 {Some(f)} else {None})
        .map(|frame| frame
            .iter()
            .copied()
            .map(|i| i as f32)
            .map(|i| Complex { re: i, im: 0f32})
            .collect::<Vec<Complex<f32>>>()
        )
        .map(|mut frame | {
            fft.process(frame.as_mut_slice());
            frame
        })
        .map(|frame| frame.iter().map(|i| i/(WINDOW_SIZE as f32).sqrt()).collect())
        .collect::<Vec<Vec<Complex<f32>>>>();
    if let Some(frame) = frames.get(0) {
        println!("{:?}", frame);
    }
    
    
    Ok(())
}
