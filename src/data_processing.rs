extern crate rand;
extern crate sprs;
use rand::{seq::SliceRandom, SeedableRng};
use rand::thread_rng;
use rand::Rng;
use rand_chacha::ChaChaRng;
//use std::borrow::Borrow;
//use std::cell::RefCell;
//use std::collections::HashMap;
use std::convert::TryInto;
//use ndarray::Array1;
//use ndarray::Array2;

use sprs::{CsMat,CsVec};

// TODO: seed is unused

fn BSC_channel(n:usize ,code: &mut Vec<usize>, proba: f64) ->(CsVec<usize>, Vec<f64>){
    let mut rng = rand::thread_rng();
    let mut received: Vec<usize> = Vec::new();
    let mut post_proba : Vec<f64> = Vec::new();
    for _i in 0..code.len(){
        if rng.gen::<f64>() < proba{
            code[_i] ^= 1;
        }
        if code[_i] == 1{
            received.push(_i);
            post_proba.push(proba);
        }else{post_proba.push(1.0-proba);}
    } 
    let data = vec![1;received.len()];
    (CsVec::new(n,received,data),post_proba)
}

fn make_matrix_regularLDPC(w_c: usize, w_r: usize, n: usize, seed: u8) -> CsMat<usize> {
    if (n * w_c) % w_r != 0 {
        panic!("number of col * weight of col must be divisible by weight of row");
    }

    // TODO: what is this used for ?
    let num_row: usize = n * w_c / w_r as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    let mut indptr : Vec<usize> = vec![0;num_row+1 as usize];
    let data: Vec<usize> = vec![0;(n*w_c as usize).try_into().unwrap()];
    let indices_copy: Vec<usize> = indices.clone();
    for i in 0..(num_row+1){
        indptr[i as usize] = i*w_r as usize;
    }
    for i in 0..(w_c-1){
        let mut temp = indices_copy.clone();
//        temp.shuffle(&mut thread_rng());
        let seed_list = [seed * i as u8; 32];
        let mut rng = ChaChaRng::from_seed(seed_list);
        temp.shuffle(&mut rng);
        indices.append(&mut temp);
    }    
    CsMat::new_csc((num_row,n),
                    indptr,
                    indices,
                    data)

}

fn input_regularLDPC(m:usize, n:usize,matrix: &CsMat<usize>, post_proba: &Vec<f64>) -> CsMat<f64>{
//    let (indptr, indices, _) = matrix.into_raw_storage();
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0;m+1];    // Need to check it if there is a logic error
    for index in 0..n{
        indptr_clone[index] = matrix.indptr().outer_inds_sz(index).start;
    }    
    indptr_clone[n] = nnz;
    let mut indices: Vec<usize> = Vec::new();
    for index in matrix.indices(){
        indices.push(*index);
    }
    let mut data = Vec::new();
    for _i in matrix.indices().iter(){
        let temp = post_proba[*_i]/(1.0-post_proba[*_i]);
        data.push(temp.ln());
    }
    CsMat::new((m,n),
                indptr_clone,
                indices,
                data)
} 

pub fn MessagePassing(matrix: &mut CsMat<usize>,syndrome: CsVec<usize>,post_proba:Vec<f64>,number_of_iter: usize ) 
                -> Option<CsVec<usize>> {
    let (m,n) = matrix.shape();
    let mut success = false;
    let mut matrix_input = input_regularLDPC(m, n, matrix, &post_proba);
    let mut syndrome_vec : Vec<usize> = vec![0;m];
    for i in syndrome.indices(){
        syndrome_vec[*i] = 1;
    }
    for _i in 0..number_of_iter{
        horizontal_run(&mut matrix_input, &syndrome_vec);
        let code_word = vertical_run(&mut matrix_input, &post_proba, n, m);
        success = verification(matrix, &code_word, &syndrome);
        match success{
            true => return Some(code_word),
            false => continue
        }
    }
    return None;
}

fn horizontal_run(matrix: &mut CsMat<f64>, syndrome: &Vec<usize>) {
    let (m,_) = matrix.shape();  
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0;m+1];
    for index in 0..m{
        indptr_clone[index] = matrix.indptr().outer_inds_sz(index).start;
    }    
    indptr_clone[m] = nnz;
    let data = matrix.data_mut();
    for index in 0..m{
        let temp: f64 = data[indptr_clone[index]..indptr_clone[index+1]].iter().product();
        for _i in indptr_clone[index]..indptr_clone[index+1]{
            let mut temp1 : f64 = 0.0;        
            if syndrome[index] == 1{
                temp1 = temp/data[_i];
            }else{
                temp1 = -temp/data[_i];
            }
            let temp2 = (1.0+temp1)/(1.0-temp1);
            data[_i] = temp2.ln();
        }
    }
}

fn vertical_run(matrix: &mut CsMat<f64>, post_proba: &Vec<f64>, n: usize,m:usize) -> CsVec<usize>{
    let mut indices = Vec::new();

    let mut matrix_temp = matrix.to_csc();
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0;m+1];
    for index in 0..n{
        indptr_clone[index] = matrix_temp.indptr().outer_inds_sz(index).start;
    }    
    indptr_clone[n] = nnz;
    let data = matrix_temp.data_mut();
    for index in 0..n{
        let mut temp: f64 = data[indptr_clone[index]..indptr_clone[index+1]].iter().sum();
        temp += post_proba[index];
        for _i in indptr_clone[index]..indptr_clone[index+1]{        
            data[_i] = temp - data[_i]
        }
        if temp <= 0.0{
            indices.push(index);
        }
    }
    let data_vec = vec![1;indices.len()];
    let matrix_temp = matrix_temp.to_csr();
    let data = matrix.data_mut();
    for datum_ind in 0..nnz{
        data[datum_ind] = matrix_temp.data()[datum_ind];
    }
    CsVec::new(n, indices,data_vec)
}

fn verification(matrix0: &CsMat<usize>, code_word: &CsVec<usize>, syndrome: &CsVec<usize>) -> bool {
    // Maybe there's a better way to do this, but here's what I used:
    // https://stackoverflow.com/questions/66925648/how-do-i-create-a-two-dimensional-array-from-a-vector-of-vectors-with-ndarray
    if matrix0 * code_word == *syndrome{
        true
    }else{false}
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
mod tests{
    use super::*;
    #[test]
    fn demo() {
//        println!("This is inside a test");
        let n = 1000000;
        let w_c = 4;
        let w_r = 8;
        let crossover_proba = 0.02;
        let seed =10;
        let mut original_code_word : Vec<usize> = vec![0;n];
        let mut matrix = make_matrix_regularLDPC(w_c, w_r, n, seed);
        let (received, post_proba) = BSC_channel(n, &mut original_code_word, crossover_proba);
        let syndrome = &matrix * &received;
        match MessagePassing(&mut matrix, syndrome, post_proba, 60){
            Some(_value) => {
                assert_eq!(2,2);
            }
            None => assert_eq!(2,3)
        }
    }
}