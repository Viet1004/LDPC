mod data_processing;
use sprs::CsVec;
use rand::thread_rng;
use rand::seq::SliceRandom;

use std::time::{Duration, Instant};

fn main() {
    let n = 120000;
    let w_c = 4;
    let w_r = 8;
    let n = w_r * (n / w_r);
    let m = n * w_c/w_r;
    println!("n={}", n);
    let crossover_proba = 0.05;
    let seed = 10;
    let mut original_code_word: Vec<usize> = vec![0; n];
    let mut original_compact_form: Vec<usize> = vec![];
    let mut num_one = 0;
    let mut rng = thread_rng();
    let choices: [usize;2] = [0,1];
    for i in 0..n{
      original_code_word[i] = *choices.choose(&mut rng).unwrap();
      if original_code_word[i] == 1{
         original_compact_form.push(i);
         num_one += 1;
      }
    }  
//    println!("Original codeword is {:?}", original_code_word); 
    let original_compact_form_clone = original_compact_form.clone();
    let sparse_vec = CsVec::new(n, original_compact_form, vec![1; num_one]);
    let mut time0 = Instant::now();
    let (received, post_proba) = data_processing::bsc_channel(n, &original_code_word, crossover_proba);
    let mut time1 = Instant::now();
    println!("time to generate receive vector: {:?}", time1.saturating_duration_since(time0));
    time0 = Instant::now();
    let mut matrix = data_processing::make_matrix_regular_ldpc(w_c, w_r, n, seed);
    time1 = Instant::now();
    println!("time to generate matrix: {:?}", time1.saturating_duration_since(time0));
    let syndrome = &matrix * &sparse_vec;
    let mut new_data : Vec<usize> = Vec::new();
    let mut new_indices : Vec<usize> = Vec::new();
    for i in 0..syndrome.data().len(){
        if syndrome.data()[i] % 2 == 1{
            new_indices.push(syndrome.indices()[i]);
            new_data.push(1);
        }
    }
    let syndrome = CsVec::new(m,new_indices,new_data);
    time0 = Instant::now();
    match data_processing::message_passing(&mut matrix, syndrome, post_proba, 60) {
        Some(value) => {
            println!("That is great but not quite!");
            assert_eq!(value.indices(), original_compact_form_clone);
        }
        None => {
            println!("Sorry mate!")
        }
    }
    time1 = Instant::now();
    println!("time to (not) decode: {:?}", time1.saturating_duration_since(time0));
//    data_processing::message_passing_test();
    //   println!("Hello");
}
