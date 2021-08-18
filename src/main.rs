mod data_processing;
use sprs::CsVec;

use std::time::{Duration, Instant};

fn main() {
    let n = 840000;
    let w_c = 3;
    let w_r = 6;
    let n = w_r * (n / w_r);
    println!("n={}", n);
    let crossover_proba = 0.05;
    let seed = 10;
    let mut original_code_word: Vec<usize> = vec![0; n];
    let sparse_vec = CsVec::new(n, vec![0; 0], vec![1; 0]);
    let mut time0 = Instant::now();
    let (received, post_proba) = data_processing::bsc_channel(n, &mut original_code_word, crossover_proba);
    let mut time1 = Instant::now();
    println!("time to generate receive vector: {:?}", time1.saturating_duration_since(time0));

    //   let a = CsMat::new_csc((9,12),
    //                     vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
    //                     vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 7, 3, 5, 9, 11, 1, 6, 8, 10, 2, 4, 6, 9, 5, 7, 10, 11, 0, 1, 3, 8],
    //                     vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    //   println!("Hello!");
    time0 = Instant::now();
    let mut matrix = data_processing::make_matrix_regular_ldpc(w_c, w_r, n, seed);
    time1 = Instant::now();
    println!("time to generate matrix: {:?}", time1.saturating_duration_since(time0));

    //   println!("matrix indice: {:?}", matrix);
    let syndrome = &matrix * &sparse_vec;
    //   println!("syndrome: {:?}",syndrome.indices());
    //   println!("received: {:?}", received);
    time0 = Instant::now();
    match data_processing::message_passing(&mut matrix, syndrome, post_proba, 60) {
        Some(value) => {
            println!("That is great but not quite!");
            //         assert_eq!(value.indices(), original_code_word)
        }
        None => {
            println!("Sorry mate!")
        }
    }
    time1 = Instant::now();
    println!("time to (not) decode: {:?}", time1.saturating_duration_since(time0));

    //   println!("Hello");
}
