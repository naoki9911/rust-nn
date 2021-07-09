extern crate mnist;
extern crate ndarray;

use mnist::{Mnist, MnistBuilder};
use ndarray::{s, Array, Array2};

mod optimizer;
use optimizer::Adam;

mod layer;
use layer::{Affine, ReLU, Softmax};

mod utils;
use utils::*;

mod model;
use model::Model;

fn main() {
    let (rows, cols) = (28, 28);

    let Mnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .base_path("mnist")
        .label_format_digit()
        .finalize();

    let trn_lbl: Vec<usize> = trn_lbl.into_iter().map(|v| v as usize).collect();
    let tst_lbl: Vec<usize> = tst_lbl.into_iter().map(|v| v as usize).collect();

    let trn_size = trn_img.len() / (rows * cols);
    let tst_size = tst_img.len() / (rows * cols);
    println!("trn_size:{} tst_size:{}", trn_size, tst_size);
    let trn_img: Vec<f32> = trn_img.into_iter().map(|v| v as f32).collect();
    let trn_img: Array2<f32> = Array::from(trn_img).into_shape((trn_size, rows*cols)).unwrap();
    let trn_img: Array2<f32> = trn_img.reversed_axes() / 255.0;
    let tst_img: Vec<f32> = tst_img.into_iter().map(|v| v as f32).collect();
    let tst_img: Array2<f32> = Array::from(tst_img).into_shape((tst_size, rows * cols)).unwrap();
    let tst_img: Array2<f32> = tst_img.reversed_axes() / 255.0;

    let mut model = Model::new();
    model.add_layer(Box::new(Affine::<Adam>::new(rows*cols, 1000)));
    model.add_layer(Box::new(ReLU::new(1000, 1000)));
    model.add_layer(Box::new(Affine::<Adam>::new(1000, 10)));
    model.add_layer(Box::new(Softmax::new(10, 10)));
    model.train(&trn_img, &trn_lbl, &tst_img, &tst_lbl);

    let ans = model.eval(&tst_img.slice(s![.., 0..1]).into_owned());
    show_img(
        &tst_img
            .slice(s![.., 0..1])
            .into_owned()
            .into_shape((28, 28))
            .unwrap(),
    );
    println!("Predicted:{}", ans[0]);
}
