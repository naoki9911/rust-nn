extern crate ndarray;
extern crate ndarray_rand;

use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub trait Optimizer {
    fn new(out_dim: usize, in_dim: usize) -> Self;

    fn update(&mut self, grad: &Array2<f32>);

    fn get(&self) -> &Array2<f32>;
}

#[derive(Clone)]
pub struct SGD {
    var: Array2<f32>,
}

impl Optimizer for SGD {
    fn new(out_dim: usize, in_dim: usize) -> Self {
        Self {
            var: Array::random(
                (out_dim, in_dim),
                Uniform::new(-0.5 / out_dim as f32, 0.5 / out_dim as f32),
            ),
        }
    }

    fn update(&mut self, grad: &Array2<f32>) {
        self.var = self.var.clone() - 0.01 * grad;
    }

    fn get(&self) -> &Array2<f32> {
        return &self.var;
    }
}

#[derive(Clone)]
pub struct MomentumSGD {
    var: Array2<f32>,
    delta: Array2<f32>,
}

impl Optimizer for MomentumSGD {
    fn new(out_dim: usize, in_dim: usize) -> Self {
        Self {
            var: Array::random(
                (out_dim, in_dim),
                Uniform::new(-0.5 / out_dim as f32, 0.5 / out_dim as f32),
            ),
            delta: Array::random(
                (out_dim, in_dim),
                Uniform::new(-0.5 / out_dim as f32, 0.5 / out_dim as f32),
            ),
        }
    }

    fn update(&mut self, grad: &Array2<f32>) {
        self.delta = 0.9 * &self.delta - 0.01 * grad;
        self.var = &self.var + &self.delta;
    }

    fn get(&self) -> &Array2<f32> {
        return &self.var;
    }
}
