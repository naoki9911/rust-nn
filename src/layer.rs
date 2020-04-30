extern crate ndarray;
use ndarray::{Array2, Axis};

use crate::optimizer::Optimizer;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grad: &Array2<f32>) -> Array2<f32>;
}

#[derive(Clone)]
pub struct Affine<T: Optimizer> {
    input: Array2<f32>,
    weight: T,
    baias: T,
}

impl<T: Optimizer> Affine<T> {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            input: Array2::zeros((1, 1)),
            weight: T::new(out_dim, in_dim),
            baias: T::new(out_dim, 1),
        }
    }
}

impl<T: Optimizer> Layer for Affine<T> {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.input = input.clone();
        let res = self.weight.get().dot(input) + self.baias.get();
        return res;
    }

    fn backward(&mut self, grad: &Array2<f32>) -> Array2<f32> {
        let grad_w = grad.dot(&self.input.clone().reversed_axes());
        let grad_b = grad
            .sum_axis(Axis(1))
            .into_shape(self.baias.get().dim())
            .unwrap();
        let grad_x = self.weight.get().clone().reversed_axes().dot(grad);

        self.weight.update(&grad_w);
        self.baias.update(&grad_b);

        return grad_x;
    }
}

#[derive(Clone)]
pub struct ReLU {
    input: Array2<f32>,
}

impl ReLU {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            input: Array2::zeros((1, 1)),
        }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.input = input.clone();
        return input.mapv(|v| if v > 0.0 { v } else { 0.0 });
    }

    fn backward(&mut self, grad: &Array2<f32>) -> Array2<f32> {
        let masked = self.input.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });

        return grad * &masked;
    }
}

#[derive(Clone)]
pub struct Softmax {
    input_dimension: usize,
    input: Array2<f32>,
    output: Array2<f32>,
}

impl Softmax {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Self {
            input_dimension: in_dim,
            input: Array2::zeros((1, 1)),
            output: Array2::zeros((1, 1)),
        }
    }
}

impl Layer for Softmax {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let batch_size = input.dim().1;
        self.input = input.clone();
        let max = input
            .fold_axis(Axis(0), f32::MIN, |&p, &u| f32::max(p, u))
            .into_shape((1, batch_size))
            .unwrap();
        let diff = (input - &max).mapv(|v| v.exp());
        let sum = diff.sum_axis(Axis(0)).into_shape((1, batch_size)).unwrap();
        self.output = &diff / &sum;
        return self.output.clone();
    }

    fn backward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let batch_size = input.dim().1 as f32;
        return (&self.output - input) / batch_size;
    }
}
