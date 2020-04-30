extern crate ndarray;
extern crate ndarray_stats;

use ndarray::Array2;
use ndarray_stats::QuantileExt;

use ansi_term::Colour::RGB;
pub fn ans_matrix(input: &Vec<usize>, input_dimension: usize) -> Array2<f32> {
    let batch_size = input.len();
    let mut ret = Array2::<f32>::zeros((input_dimension, batch_size));
    let mut x = 0;
    for mut column in ret.gencolumns_mut() {
        let ans = one_hot_vector(input[x], input_dimension);

        for y in 0..input_dimension {
            column[y] = ans[y];
        }

        x = x + 1;
    }
    return ret;
}

pub fn one_hot_vector(ans: usize, size: usize) -> Vec<f32> {
    let mut ret: Vec<f32> = vec![0.0; size];
    ret[ans] = 1.0;
    return ret;
}

pub fn cross_entroy(
    train: &Array2<f32>,
    ans: &Vec<usize>,
    input_dimension: usize,
    batch_size: usize,
) -> f32 {
    let ans_m = ans_matrix(ans, input_dimension);
    let delta = 0.00000001;
    let log_y = (train + delta).mapv(|v| v.ln());
    let res = -ans_m * log_y;
    return res.sum() / (batch_size as f32);
}

pub fn get_ans(res: &Array2<f32>) -> Vec<usize> {
    let mut ret = Vec::<usize>::new();
    for c in res.gencolumns() {
        let max = c.argmax().unwrap();
        ret.push(max);
    }
    return ret;
}

pub fn show_img(img: &Array2<f32>) {
    for y in 0..28 {
        for x in 0..28 {
            let img_bit = img.get((y, x));
            if let Some(v) = img_bit {
                let pixel_str = format!(
                    "{}",
                    RGB(
                        255 - (*v * 255.0) as u8,
                        255 - (*v * 255.0) as u8,
                        255 - (*v * 255.0) as u8
                    )
                    .paint("■")
                );
                print!("{}", pixel_str);
            }
        }
        println!("");
    }
}
