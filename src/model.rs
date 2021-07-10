extern crate ndarray;

use crate::layer::Layer;
use crate::utils::{ans_matrix, cross_entroy, get_ans};
use ndarray::{s, Array2};

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            layers: Vec::<Box<dyn Layer>>::new(),
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn train(
        &mut self,
        trn_img: &Array2<f32>,
        trn_lbl: &Vec<usize>,
        tst_img: &Array2<f32>,
        tst_lbl: &Vec<usize>,
    ) {
        let batch_size = 100;
        let trn_size = trn_img.dim().1;
        let tst_size = tst_img.dim().1;
        let batch_num = trn_size / batch_size;
        for x in 0..10 {
            let mut loss_sum = 0.0;
            for batch in 0..batch_num {
                let mut res: Array2<f32> = trn_img
                    .slice(s![.., batch * batch_size..(batch + 1) * batch_size])
                    .into_owned();
                for l in &mut self.layers {
                    res = l.forward(&res, true);
                }
                loss_sum += cross_entroy(
                    &res,
                    &trn_lbl[batch * batch_size..(batch + 1) * batch_size].to_vec(),
                    10,
                    100,
                );
                res = ans_matrix(
                    &trn_lbl[batch * batch_size..(batch + 1) * batch_size].to_vec(),
                    10,
                );
                for l in self.layers.iter_mut().rev() {
                    res = l.backward(&res);
                }
            }
            let mut res: Array2<f32> = tst_img.clone();
            for l in &mut self.layers {
                res = l.forward(&res, false);
            }
            let ans = get_ans(&res);
            let mut ans_sum = 0;
            for x in 0..ans.len() {
                if ans[x] == tst_lbl[x] {
                    ans_sum += 1;
                }
            }
            println!(
                "epoch {}/{} trn_loss: {} tst_ans: {}",
                x + 1,
                10,
                loss_sum / batch_num as f32,
                ans_sum as f32 / tst_size as f32
            );
        }
    }

    pub fn eval(&mut self, img: &Array2<f32>) -> Vec<usize> {
        let mut res: Array2<f32> = img.clone();
        for l in &mut self.layers {
            res = l.forward(&res, false);
        }
        let ans = get_ans(&res);

        return ans;
    }
}
