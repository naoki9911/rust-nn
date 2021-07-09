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

#[derive(Clone)]
pub struct Adam {
    var: Array2<f32>,
    m: Array2<f32>,
    v: Array2<f32>,
    m_h: Array2<f32>,
    v_h: Array2<f32>,
    beta_1_t:f32,
    beta_2_t:f32
}

impl Optimizer for Adam {

    fn new(out_dim: usize, in_dim: usize) -> Self {
        const BETA_1: f32 = 0.9;
        const BETA_2: f32 = 0.999;
        Self {
            var: Array::random(
                (out_dim, in_dim),
                Uniform::new(-0.5 / out_dim as f32, 0.5 / out_dim as f32),
            ),
            m : Array2::zeros((out_dim, in_dim)),
            v : Array2::zeros((out_dim, in_dim)),
            m_h : Array2::zeros((out_dim, in_dim)),
            v_h : Array2::zeros((out_dim, in_dim)),
            beta_1_t : BETA_1,
            beta_2_t : BETA_2,
        }
    }

    fn update(&mut self, grad: &Array2<f32>) {
        const ALPHA: f32 = 0.001;
        const BETA_1: f32 = 0.9;
        const BETA_2: f32 = 0.999;
        const EPSILON: f32 = 1e-8;
        self.m = BETA_1 * &self.m + (1.0 - BETA_1)*grad;
        self.v = BETA_2 * &self.v + (1.0 - BETA_2)*grad*grad;
        self.m_h = &self.m/(1.0 - self.beta_1_t);
        self.v_h = &self.v/(1.0 - self.beta_1_t);
        self.beta_1_t = &self.beta_1_t * BETA_1;
        self.beta_2_t = &self.beta_2_t * BETA_2;
        let v_h_sqrt : Array2<f32> = &self.m_h/(&self.v_h.mapv(|v| v.sqrt() + EPSILON));
        self.var = &self.var - &v_h_sqrt.mapv(|v| v * ALPHA);
    }

    fn get(&self) -> &Array2<f32> {
        return &self.var;
    }
}