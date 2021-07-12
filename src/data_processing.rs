use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;

use ndarray::Array1;
use ndarray::Array2;

pub enum SuccessOrFail {
    Success(Vec<i8>),
    Fail(bool),
}

struct Data {
    neighbor_col: Vec<(usize, usize)>,
    neighbor_row: Vec<(usize, usize)>,
    q_0: f32,
    q_1: f32,
    r_0: f32,
    r_1: f32,
}

// TODO: seed is unused
fn make_matrix(w_c: u32, w_r: u32, n: u32, seed: i8) {
    if (n * w_c) % w_r != 0 {
        panic!("number of col * weight of col must be divisible by weight of row");
    }

    // TODO: what is this used for ?
    let num_row: u32 = n * w_c / w_r;
}

pub fn run(m: usize, n: usize, matrix: Vec<Vec<i8>>, proba: Vec<(f32, f32)>, check: Vec<i8>,
           number_of_iter: usize) -> SuccessOrFail {
    let mut result: Vec<i8> = vec![0; n];
    let mut dict: HashMap<(usize, usize), RefCell<Data>> = HashMap::new();
    for i in 0..m {
        for j in 0..n {
            if matrix[i][j] == 1 {
                let mut neighbor_c: Vec<(usize, usize)> = Vec::new();
                let mut neighbor_r: Vec<(usize, usize)> = Vec::new();
                for k in 0..n {
                    if matrix[i][k] == 1 && k != j {
                        neighbor_r.push((i, k));
                    }
                }
                for k in 0..m {
                    if matrix[k][j] == 1 && k != i {
                        neighbor_c.push((k, j))
                    }
                }
                let (mut q_0, mut q_1) = proba[j];
                let (mut r_0, mut r_1) = (1.0, 1.0);
                let data = Data {
                    neighbor_col: neighbor_c,
                    neighbor_row: neighbor_r,
                    q_0,
                    q_1,
                    r_0,
                    r_1,
                };
                dict.insert((i, j), RefCell::new(data));
            }
        }
    }

    for _ in 0..number_of_iter {
        horizontal_run(&mut dict);
        vertical_run(&mut dict, &proba, &mut result);
        match verification(&matrix, &result, &check) {
            true => {
                return SuccessOrFail::Success(result);
            }
            false => {
                continue;
            }
        }
    }
    SuccessOrFail::Fail(false)
}

fn horizontal_run(dict: &mut HashMap<(usize, usize), RefCell<Data>>) {
    for (l, r) in dict.keys() {
        let mut mul = 1.0f32;
        let mut value = match dict.get(&(*l, *r)) {
            None => {
                // TODO: are we sure we want to crash the program if that happens ? Should there be a more detailed error message ?
                panic!("there is no value associated with key!!!")
            }
            Some(v) => v
        };

        for (nl, nr) in &value.borrow().neighbor_row {
            // TODO: are we sure we want to unwrap here ? Eg crash if even one get fails ?
            mul *= dict.get(&(*nl, *nr)).unwrap().borrow().q_0 - dict.get(&(*nl, *nr)).unwrap().borrow().q_1;
        }

        { value.borrow_mut().r_0 = (1.0 + mul) / 2.0; }
        value.borrow_mut().r_0 = (1.0 - mul) / 2.0;
    }
}

fn vertical_run(dict: &mut HashMap<(usize, usize), RefCell<Data>>, proba: &Vec<(f32, f32)>, string: &mut Vec<i8>) {
    for (l, r) in dict.keys() {
        let mut q0: f32 = proba[*r].0;
        let mut q1: f32 = proba[*r].1;
        // TODO: check if entry is present
        for neighborC in &dict.get(&(*l, *r)).unwrap().borrow().neighbor_col {
            q0 *= dict.get(&neighborC).unwrap().borrow_mut().r_0;
            q1 *= dict.get(&neighborC).unwrap().borrow_mut().r_1;
        }
        let sum = q0 + q1;
        q0 = q0 / sum;
        q1 = q1 / sum;
        let temp = match dict.get(&(*l, *r)) {
            Some(val) => val,
            None => panic!("Cannot find value associated with key line 106 "),
        };

        { temp.borrow_mut().q_0 = q0; }
        { temp.borrow_mut().q_1 = q1; }

        q0 = q0 * ((*dict).get(&(*l, *r)).unwrap().borrow().q_0);
        q1 = q1 * ((*dict).get(&(*l, *r)).unwrap().borrow().q_1);
        if q1 >= q0 {
            string[*r] = 1;
        } else {
            string[*r] = 0;
        }
    }
}

fn verification(matrix: &Vec<Vec<i8>>, string: &Vec<i8>, check: &Vec<i8>) -> bool {
    // Maybe there's a better way to do this, but here's what I used:
    // https://stackoverflow.com/questions/66925648/how-do-i-create-a-two-dimensional-array-from-a-vector-of-vectors-with-ndarray
    let mut data = Vec::new();
    let ncols = matrix.first().map_or(0, |row| row.len());
    let mut nrows = 0;
    for i in 0..matrix.len() {
        data.extend_from_slice(matrix[i].as_slice());
        nrows += 1;
    }

    let matrix = Array2::from_shape_vec((nrows, ncols), data.iter().map(|e| *e).collect()).unwrap();

    // TODO: I'm not sure about this (the fact that it is called string makes me doubtful)
    let string = Array1::from_vec(string.to_vec());
    let check = Array1::from_vec(check.to_vec());

    let res: Array1<i8> = matrix.dot(&string);
    for i in 0..res.len() {
        if res[i] != check[i] { return false; } else { continue; }
    }
    true
}
//#[cfg(test)]
//mod tests{
//    use super::*;
//    #[test]
//    fn horizontal_test(){
//        assert_eq!(vec![vec![0,1]], horizontal_run());
//    }
//}
