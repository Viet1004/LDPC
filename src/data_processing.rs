extern crate rand;
extern crate sprs;
use rand::{seq::SliceRandom, SeedableRng};
use rand::thread_rng;
use rand::Rng;
use rand_chacha::ChaChaRng;
//use std::borrow::Borrow;
//use std::cell::RefCell;
//use std::collections::HashMap;
//use std::convert::TryInto;
//use ndarray::Array1;
//use ndarray::Array2;

use sprs::{CsMat,CsVec};

// TODO: seed is unused

pub fn bsc_channel(n:usize ,code: &Vec<usize>, proba: f64) ->(CsVec<usize>, Vec<f64>){
    let mut rng = rand::thread_rng();
    let mut received: Vec<usize> = Vec::new();
    let mut post_proba : Vec<f64> = Vec::new();
    let mut code_clone = code.clone();
    for _i in 0..code.len(){
        if rng.gen::<f64>() < proba{
            code_clone[_i] ^= 1;
        }
        if code_clone[_i] == 1{
            received.push(_i);
            post_proba.push(proba);
        }else{post_proba.push(1.0-proba);}
    } 
    let data = vec![1;received.len()];
    (CsVec::new(n,received,data),post_proba)
}

pub fn make_matrix_regular_ldpc(w_c: usize, w_r: usize, n: usize, seed: u8) -> CsMat<usize> {
    // TODO: what is this used for ?
    let num_row: usize = n * w_c / w_r as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    let mut indptr : Vec<usize> = vec![0;num_row+1];
    let data: Vec<usize> = vec![1;n*w_c];
    let indices_copy: Vec<usize> = indices.clone();
    for i in 0..(num_row+1){
        indptr[i as usize] = i*w_r;
    }
    for i in 0..(w_c-1){
        let mut temp = indices_copy.clone();
//        temp.shuffle(&mut thread_rng());
        let seed_list = [seed * i as u8; 32];
        let mut rng = ChaChaRng::from_seed(seed_list);
        temp.shuffle(&mut rng);
        for j in 0..(n/w_r){
            &temp[j*w_r..(j+1)*w_r].sort_unstable();
            
        }
        indices.append(&mut temp);
    }    
    CsMat::new((num_row,n),
                    indptr,
                    indices,
                    data)
}

fn input_regular_ldpc(m:usize, n:usize,matrix: &CsMat<usize>, post_proba: &Vec<f64>) -> CsMat<f64>{
//    let (indptr, indices, _) = matrix.into_raw_storage();
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0;m+1];    // Need to check it if there is a logic error
    for index in 0..m{
        indptr_clone[index] = matrix.indptr().outer_inds_sz(index).start;
    }
//    println!("{:?}",indptr_clone);    
    indptr_clone[m] = nnz;
    let mut indices: Vec<usize> = Vec::new();
    for index in matrix.indices(){
        indices.push(*index);
    }
    let mut data = Vec::new();
    for _i in matrix.indices().iter(){
        let temp = post_proba[*_i]/(1.0-post_proba[*_i]);
        data.push(temp.ln());
    }
//    println!("indptr_clone{:?}", indptr_clone);
    CsMat::new((m,n),
                indptr_clone,
                indices,
                data)
} 

pub fn message_passing(matrix: &mut CsMat<usize>,syndrome: CsVec<usize>,post_proba:Vec<f64>,number_of_iter: usize ) 
                -> Option<CsVec<usize>> {
    let (m,n) = matrix.shape();
    let mut success = false;
    let mut matrix_input = input_regular_ldpc(m, n, matrix, &post_proba);
    let mut syndrome_vec : Vec<usize> = vec![0;m];
    let mut post_proba_log:Vec<f64> = vec![0.0;n];
    for i in 0..n{
        post_proba_log[i] = (post_proba[i]/(1.0-post_proba[i])).ln();
    }
//    println!("Post proba is {:?}", post_proba_log);
//    println!("Matrix input is {:?}", matrix_input);
    
    for i in syndrome.indices(){
        syndrome_vec[*i] = 1;
//        println!("i is {}", *i);
    }
    for _i in 0..number_of_iter{
        horizontal_run(&mut matrix_input, &syndrome_vec);
//        println!("After horizontal run: {:?}", matrix_input);
        let code_word = vertical_run(&mut matrix_input, &post_proba_log, n, m);
//        println!("After vertical run: {:?}", matrix_input);
//        println!("codeword after modification: {:?}", code_word);
//        println!("syndrome: {:?}", syndrome);
        success = verification(matrix, &code_word, &syndrome,m);
        match success{
            true => {
                println!("Number of iteration :{}",_i);
                return Some(code_word);
            }
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
//    println!("Before horizontal run: {:?}", data);
    for index in 0..nnz{
        data[index] = (data[index]/2.0).tanh();
    }
    for index in 0..m{
        let temp: f64 = data[indptr_clone[index]..indptr_clone[index+1]].iter().product();
        for _i in indptr_clone[index]..indptr_clone[index+1]{
            let temp1 : f64;        
            if syndrome[index]%2 == 1{
                temp1 = -temp/data[_i];
            }else{
                temp1 = temp/data[_i];
            }
            let temp2 = (1.0+temp1)/(1.0-temp1);
            data[_i] = temp2.ln();
        }
    }
//    println!("After Horizontal run: {:?}", data);
}

fn vertical_run(matrix: &mut CsMat<f64>, post_proba_log: &Vec<f64>, n: usize,m:usize) -> CsVec<usize>{
    let mut indices: Vec<usize> = Vec::new();

    let mut matrix_temp = matrix.to_csc();
    let nnz = matrix.nnz();
    let mut indptr_clone: Vec<usize> = vec![0;n+1];
    for index in 0..n{
        indptr_clone[index] = matrix_temp.indptr().outer_inds_sz(index).start;
    }    
    indptr_clone[n] = nnz;
    let data = matrix_temp.data_mut();
//    println!("Data before vertical run: {:?}", data);
    for index in 0..n{
        let mut temp: f64 = data[indptr_clone[index]..indptr_clone[index+1]].iter().sum();
        temp += post_proba_log[index];
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
//    println!("data after vertical run:{:?}", data);
    CsVec::new(n, indices,data_vec)
}

fn verification(matrix0: &CsMat<usize>, code_word: &CsVec<usize>, syndrome: &CsVec<usize>,m:usize) -> bool {
    // Maybe there's a better way to do this, but here's what I used:
    // https://stackoverflow.com/questions/66925648/how-do-i-create-a-two-dimensional-array-from-a-vector-of-vectors-with-ndarray
    let temp = matrix0 * code_word;
    let mut new_data : Vec<usize> = Vec::new();
    let mut new_indices : Vec<usize> = Vec::new();
    for i in 0..temp.indices().len(){
        if temp.data()[i] % 2 == 1{
            new_indices.push(temp.indices()[i]);
            new_data.push(1);
        }
    }
    let trial = CsVec::new(m,new_indices, new_data);
//    println!("Trial is : {:?}", trial);
    if trial == *syndrome{
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

pub fn message_passing_test() {
    let n = 12;
    let m = 9;
    let w_c = 3;
    let w_r = 4;
    let crossover_proba = 0.2;
    let mut original_code_word = vec![0,0,1,0,0,0,1,0,0,0,0,1];
    let mut original_compact_form :Vec<usize> = Vec::new();
    let mut num_one: usize = 0;
    let mut rng = thread_rng();
    let choices: [usize;2] = [0,1];
    for i in 0..n{
      original_code_word[i] = *choices.choose(&mut rng).unwrap();
      if original_code_word[i] == 1{
         original_compact_form.push(i);
         num_one += 1;
      }
    }

    let sparse_vec = CsVec::new(12,original_compact_form, vec![1;num_one]);
    println!("{:?}", sparse_vec);
    let (received, post_proba) = bsc_channel(n, &original_code_word, crossover_proba);
    println!("After bsc channel: {:?} \n {:?}",received, post_proba);
    let mut matrix = CsMat::new((9,12),
                                    vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
                                    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 7, 3, 5, 9, 11, 1, 6, 8, 10, 2, 4, 6, 9, 5, 7, 10, 11, 0, 1, 3, 8],
                                    vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    let post_proba:Vec<f64> = vec![0.8,0.8,0.2,0.8,0.8,0.8,0.2,0.8,0.8,0.8,0.8,0.2];
    let mut post_proba_log:Vec<f64> = vec![0.0;n];
    for i in 0..n{
        post_proba_log[i] = (post_proba[i]/(1.0-post_proba[i])).ln();
    }
    let syndrome_tests: CsVec<usize> = CsVec::new(9,vec![0,1,2,3,4,5,7], vec![1,1,1,1,1,1,1]);
    
    let mut matrix_input = input_regular_ldpc(m, n, &matrix, &post_proba);
    let post_process_code_word = message_passing(&mut matrix, syndrome_tests, post_proba, 1);

    //    let syndrome = vec![0,1,0,1,0,0,0,0,1];
//    horizontal_run(&mut matrix_input, &syndrome);  
//    println!("{:?}", matrix_input);  
//    let _ = vertical_run(&mut matrix_input, &post_proba_log, n, m);
//    println!("{:?}",matrix_input);
//  [0,1,0,0,1,1,0,1,0,0,0,1]
    let vec_tests: CsVec<usize> = CsVec::new(12,vec![2,6,11], vec![1,1,1]);
//    let test_product = &matrix *&vec_tests;
//    println!("Test product is {:?}", test_product);
    let syndrome_tests: CsVec<usize> = CsVec::new(9,vec![0,1,2,3,4,5,7], vec![1,1,1,1,1,1,1]);
    println!("{:?}",verification(&matrix, &vec_tests, &syndrome_tests,m));
     
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn horizontal_run_test() {
        let n = 12;
        let m = 9;
        let w_c = 3;
        let w_r = 4;
        let mut matrix = CsMat::new_csc((9,12),
        vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 7, 3, 5, 9, 11, 1, 6, 8, 10, 2, 4, 6, 9, 5, 7, 10, 11, 0, 1, 3, 8],
        vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
        let post_proba = vec![0.8,0.8,0.2,0.8,0.8,0.8,0.2,0.8,0.8,0.8,0.8,0.2];
        let mut matrix_input = input_regular_ldpc(m, n, &matrix, &post_proba);
        let syndrome = vec![0,1,0,1,0,0,0,0,1,1,0,0];
        horizontal_run(&mut matrix_input, &syndrome);  
        println!("{:?}", matrix_input);     
    }
}