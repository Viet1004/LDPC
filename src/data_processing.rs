extern crate rand;
extern crate sprs;
//use rand::thread_rng;
use rand::Rng;
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaChaRng;
//use std::borrow::Borrow;
//use std::cell::RefCell;
//use std::collections::HashMap;
//use std::convert::TryInto;
//use ndarray::Array1;
//use ndarray::Array2;

//use sprs::{CsMat, CsVec};
use sprs::CsMat;

// TODO: seed is unused

// added functions
pub fn csmat_dot(mat: &CsMat<usize>, vec: &Vec<usize>) -> Vec<usize> {
    let indptr = mat.indptr();
    let indices = mat.indices();
    let (m, n) = mat.shape();
    let mut mult = Vec::with_capacity(m);

    for i in 0..m {
        //let r = indices[indptr[i]..indptr[i + 1]];
        //println!("r: {:?}", indptr.outer_inds_sz(i));
        let range = indptr.outer_inds_sz(i);
        let mut res = 0;
        for j in range {
            res += vec[indices[j]];
        }
        mult.push(res % 2);
    }

    //println!("mult is {:?}", mult);
    mult
}

//

// pub fn bsc_channel(n: usize, code: &Vec<usize>, proba: f64) -> (CsVec<usize>, Vec<f64>) {
//     let mut rng = rand::thread_rng();
//     let mut received: Vec<usize> = Vec::new();
//     let mut post_proba: Vec<f64> = Vec::new();
//     let mut code_clone = code.clone();
//     for _i in 0..code.len() {
//         if rng.gen::<f64>() < proba {
//             code_clone[_i] ^= 1;
//         }
//         if code_clone[_i] == 1 {
//             received.push(_i);
//             post_proba.push(proba);
//         } else {
//             post_proba.push(1.0 - proba);
//         }
//     }
//     let data = vec![1; received.len()];
//     (CsVec::new(n, received, data), post_proba)
// }

pub fn bsc_channel(n: usize, codeword: &Vec<usize>, proba: f64) -> (Vec<usize>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let n = codeword.len();
    let mut received: Vec<usize> = codeword.clone();
    for i in 0..n {
        if rng.gen::<f64>() < proba {
            received[i] ^= 1;
        }
    }
    let post_proba = received.iter().map(|&x| if x == 1 { proba } else { 1.0 - proba }).collect();

    //let data = vec![1; received.len()];
    //(CsVec::new(n, received, data), post_proba)
    (received, post_proba)
}

pub fn make_matrix_regular_ldpc(w_c: usize, w_r: usize, n: usize, seed: u8) -> CsMat<usize> {
    if (n * w_c) % w_r != 0 {
        panic!("number of col * weight of col must be divisible by weight of row");
    }

    // TODO: what is this used for ?
    let num_row: usize = n * w_c / w_r as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    let mut indptr: Vec<usize> = vec![0; num_row + 1];
    let data: Vec<usize> = vec![1; n * w_c];
    let indices_copy: Vec<usize> = indices.clone();
    for i in 0..(num_row + 1) {
        indptr[i as usize] = i * w_r;
    }
    for i in 0..(w_c - 1) {
        let mut temp = indices_copy.clone();
        //        temp.shuffle(&mut thread_rng());
        let seed_list = [seed * i as u8; 32];
        let mut rng = ChaChaRng::from_seed(seed_list);
        temp.shuffle(&mut rng);
        for j in 0..(n / w_r) {
            &temp[j * w_r..(j + 1) * w_r].sort_unstable();
        }
        indices.append(&mut temp);
    }
    CsMat::new((num_row, n), indptr, indices, data)
}

fn input_regular_ldpc(m: usize, n: usize, matrix: &CsMat<usize>, post_proba: &Vec<f64>) -> CsMat<f64> {
    //    let (indptr, indices, _) = matrix.into_raw_storage();
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0; m + 1]; // Need to check it if there is a logic error
    for index in 0..m {
        indptr_clone[index] = matrix.indptr().outer_inds_sz(index).start;
    }
    //    println!("{:?}",indptr_clone);
    indptr_clone[m] = nnz;
    let mut indices: Vec<usize> = Vec::new();
    for index in matrix.indices() {
        indices.push(*index);
    }
    let mut data = Vec::new();
    for _i in matrix.indices().iter() {
        let temp = post_proba[*_i] / (1.0 - post_proba[*_i]);
        data.push(temp.ln());
    }
    //    println!("indptr_clone{:?}", indptr_clone);
    CsMat::new((m, n), indptr_clone, indices, data)
}

fn horizontal_run(matrix: &mut CsMat<f64>, syndrome: &[usize]) {
    let (m, _) = matrix.shape();
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0; m + 1];
    for index in 0..m {
        indptr_clone[index] = matrix.indptr().outer_inds_sz(index).start;
    }
    indptr_clone[m] = nnz;
    let data = matrix.data_mut();
    //    println!("Before horizontal run: {:?}", data);
    for index in 0..nnz {
        data[index] = (data[index] / 2.0).tanh();
    }
    for index in 0..m {
        let temp: f64 = data[indptr_clone[index]..indptr_clone[index + 1]].iter().product();
        for _i in indptr_clone[index]..indptr_clone[index + 1] {
            let mut temp1: f64 = 0.0;
            if syndrome[index] == 1 {
                temp1 = -temp / data[_i];
            } else {
                temp1 = temp / data[_i];
            }
            let temp2 = (1.0 + temp1) / (1.0 - temp1);
            data[_i] = temp2.ln();
        }
    }
    //    println!("After Horizontal run: {:?}", data);
}

fn vertical_run(matrix: &mut CsMat<f64>, post_proba: &Vec<f64>, n: usize, m: usize) -> Vec<usize> {
    let mut indices = Vec::new();

    let mut matrix_temp = matrix.to_csc();
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0; n + 1];
    for index in 0..n {
        indptr_clone[index] = matrix_temp.indptr().outer_inds_sz(index).start;
    }
    indptr_clone[n] = nnz;
    let data = matrix_temp.data_mut();
    //    println!("Data before vertical run: {:?}", data);
    for index in 0..n {
        let mut temp: f64 = data[indptr_clone[index]..indptr_clone[index + 1]].iter().sum();
        temp += post_proba[index];
        for _i in indptr_clone[index]..indptr_clone[index + 1] {
            data[_i] = temp - data[_i]
        }
        if temp <= 0.0 {
            indices.push(index);
        }
    }
    //let data_vec = vec![1; indices.len()];
    let matrix_temp = matrix_temp.to_csr();
    let data = matrix.data_mut();
    for datum_ind in 0..nnz {
        data[datum_ind] = matrix_temp.data()[datum_ind];
    }
    //    println!("data after vertical run:{:?}", data);
    //CsVec::new(n, indices, data_vec)
    let mut recv = vec![0; n];
    for i in 0..indices.len() {
        recv[indices[i]] = 1;
    }
    recv
}

fn verification(matrix0: &CsMat<usize>, received_vec: &Vec<usize>, syndrome: &[usize]) -> bool {
    // Maybe there's a better way to do this, but here's what I used:
    // https://stackoverflow.com/questions/66925648/how-do-i-create-a-two-dimensional-array-from-a-vector-of-vectors-with-ndarray
    // if matrix0 * received_vec == *syndrome {
    //     true
    // } else {
    //     false
    // }
    &csmat_dot(matrix0, received_vec) == syndrome
}

pub fn message_passing(matrix: &mut CsMat<usize>, syndrome: &[usize], post_proba: Vec<f64>, number_of_iter: usize) -> (bool, Vec<usize>) {
    let (m, n) = matrix.shape();
    let mut success = false;
    let mut matrix_input = input_regular_ldpc(m, n, matrix, &post_proba);
    // let mut syndrome_vec: Vec<usize> = vec![0; m];
    // for i in syndrome.indices() {
    //     syndrome_vec[*i] = 1;
    //     println!("i is {}", *i);
    // }
    let mut received_vec = vec![0; n];
    for i in 0..number_of_iter {
        horizontal_run(&mut matrix_input, &syndrome);
        received_vec = vertical_run(&mut matrix_input, &post_proba, n, m);
        success = verification(matrix, &received_vec, &syndrome);
        match success {
            true => {
                println!("nb of iteration: {:?}", i + 1);
                //return Some(received_vec);
                return (true, received_vec);
            }
            false => continue,
        }
    }

    //return None;
    (false, received_vec)
}

//#[cfg(test)]
//mod tests{
//    use super::*;
//    #[test]
//    fn horizontal_test(){
//        assert_eq!(vec![vec![0,1]], horizontal_run());
//    }
//}

#[cfg(test)]
mod tests {
    use super::*;
    //    #[test]
    //    fn demo() {
    //        let n = 1000000;
    //        let w_c = 4;
    //        let w_r = 8;
    //        let crossover_proba = 0.02;
    //        let seed =10;
    //        let mut original_code_word : Vec<usize> = vec![0;n];
    //        let mut matrix = make_matrix_regular_ldpc(w_c, w_r, n, seed);
    //        let (received, post_proba) = bsc_channel(n, &mut original_code_word, crossover_proba);
    //        let syndrome = &matrix * &received;
    //        match message_passing(&mut matrix, syndrome, post_proba, 60){
    //            Some(_value) => {
    //                assert_eq!(2,2);
    //            }
    //            None => assert_eq!(2,3)
    //        }
    //    }
}
