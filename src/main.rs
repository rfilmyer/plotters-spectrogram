use hound::WavReader;
use rustfft::{FftPlanner, num_complex::Complex};
use ndarray::{Array, Axis};
use plotters::prelude::*;
use ndarray_stats::QuantileExt;

const WINDOW_SIZE: usize = 1024;
const OVERLAP: f64 = 0.5;
const SKIP_SIZE: usize = (WINDOW_SIZE as f64 * (1f64 - OVERLAP)) as usize;

fn main() {
    let mut wav = WavReader::open("example.wav").unwrap();
    let samples = wav
        .samples()
        .collect::<Result<Vec<i16>, _>>()
        .unwrap();

    println!("Creating windows {} samples long from a timeline {} samples long, picking every {} windows for a total of {} windows.", WINDOW_SIZE, samples.len(), SKIP_SIZE, (samples.len() / SKIP_SIZE) - 1);

    // Convert to an ndarray
    // Hopefully this will keep me from messing up the dimensions
    // Mutable because the FFT takes mutable slices &[Complex<f32>]
    // let window_array = Array2::from_shape_vec((WINDOW_SIZE, windows_vec.len()), windows_vec).unwrap();

    let samples_array = Array::from(samples.clone());
    let windows = samples_array
        .windows(ndarray::Dim(WINDOW_SIZE))
        .into_iter()
        .step_by(SKIP_SIZE)
        .collect::<Vec<_>>()
        ;
    let windows = ndarray::stack(Axis(0), &windows).unwrap();
    println!("{:?}", windows);

    // So to perform the FFT on each window we need a Complex<f32>, and right now we have i16s, so first let's convert
    let mut windows = windows.map(|i| Complex::from(*i as f32));


    // get the FFT up and running
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);

    // Since we have a 2-D array of our windows with shape [WINDOW_SIZE, (num_samples / WINDOW_SIZE) - 1], we can run an FFT on every row.
    // Next step is to do something multithreaded with Rayon, but we're not cool enough for that yet.
    windows.axis_iter_mut(Axis(0))
        .for_each(|mut frame| { fft.process(frame.as_slice_mut().unwrap()); });
    
    // Finally, get the real component of those complex numbers we get back from the FFT
    let windows = windows.map(|i| i.re);

    // get some dimensions for drawing
    // The shape is in [nrows, ncols], but we want to transpose this.
    let height = *windows.shape().last().unwrap();
    let width = *windows.shape().first().unwrap();
    

    println!("Generating a {} wide x {} high image", width, height);

    let root_drawing_area = 
        BitMapBackend::new(
            "output.png", 
            (width as u32, height as u32), // width x height. Worth it if we ever want to resize the graph.
        ).into_drawing_area();

    let spectrogram_cells = root_drawing_area.split_evenly((width, height));

    let windows_scaled = windows.map(|i| i.abs()/(WINDOW_SIZE as f32));
    let highest_spectral_density = windows_scaled.max_skipnan();

    for (cell, spectral_density) in spectrogram_cells.iter().zip(windows_scaled.iter()) {
            let scaling_factor = spectral_density / highest_spectral_density;
            // let scaling_factor = scaling_factor.sqrt();
            let brightness: u8 = (255 as f32 * scaling_factor as f32).round() as u8;
            let color = RGBColor(brightness, brightness, brightness);
            cell.fill(&color).unwrap();
        };
}
