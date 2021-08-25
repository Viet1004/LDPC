mod data_processing;
//use sprs::{CsMat, CsVec};

use rand::Rng;
use std::time::Instant;

use std::env;

#[cfg(test)]
mod tests {

    use super::*;

    use sprs::{CsMat, CsMatBase, CsMatView, CsVec};
    use sprs::{CsVecBase, CsVecViewI};
    #[test]
    fn trysparse() {
        // Creating a sparse owned vector
        let owned = CsVec::new(10, vec![0, 4], vec![-4, 2]);
        // owned = [-4,0,0,0,2,0,0,0,0,0,0];
        // assert_eq!(owned.to_dense(), vec![-4, 0, 0, 0, 2, 0, 0, 0, 0, 0]); : to_dense => ArrayBase, ndarray.
        // Creating a sparse borrowing vector with `I = u16`
        let borrow = CsVecViewI::new(10, &[0_u16, 4], &[-4, 2]);
        // Creating a general sparse vector with different storage types
        let mixed = CsVecBase::new(10, &[0_u64, 4] as &[_], vec![-4, 2]);
        println!("{:?} {:?} {:?}", owned.to_dense(), borrow.to_dense(), mixed.to_dense());
    }

    #[test]
    fn trysmat() {
        // This creates an owned matrix
        let owned_matrix = CsMat::new((2, 2), vec![0, 1, 1], vec![1], vec![4_u8]);
        //[[0,4],[0,0]]
        // This creates a matrix which only borrows the elements
        let borrow_matrix = CsMatView::new((2, 2), &[0, 1, 1], &[1], &[4_u8]);
        // A combination of storage types may also be used for a
        // general sparse matrix
        let mixed_matrix = CsMatBase::new((2, 2), &[0, 1, 1] as &[_], vec![1_i64].into_boxed_slice(), vec![4_u8]);
        println!("owned_matrix: {:?}", owned_matrix.to_dense());
        // This creates an owned matrix
        let owned_matrix = CsMat::new((3, 3), vec![0, 3, 4, 6], vec![0, 1, 2, 1, 1, 2], vec![11, 12, 13, 22, 32, 33]);
        //[[11,12,13],[0,22,0],[0,32,33]]
        println!("owned_matrix: {:?}", owned_matrix.to_dense());
    }

    #[test]
    fn trymul() {
        let matrix = CsMat::new((2, 2), vec![0, 2, 4], vec![0, 1, 0, 1], vec![4, 5, 1, 3]);
        //[[4,5],[1,3]
        let vec = CsVec::new(2, vec![0, 1], vec![-4, 2]);
        //[-4.2]
        assert_eq!(&matrix * &vec, CsVec::new(2, vec![0, 1], vec![-6, 2]));
        //rec: [-6,2]
        println!("res: {:?}", &matrix * &vec);
    }

    #[test]
    fn tryeq() {
        let v1 = vec![1; 10];
        let v2 = vec![1; 10];
        if &v1 == &v2 {
            println!("true");
        } else {
            println!("false");
        }

        let v1 = vec![1; 10];
        let v2 = vec![0; 10];
        if &v1 == &v2 {
            println!("true");
        } else {
            println!("false");
        }
    }

    use rand::seq::SliceRandom;
    use rand::Rng;
    #[test]
    fn tryrand() {
        let mut rng = rand::thread_rng();
        let n = 10;
        let time0 = Instant::now();
        let mut y = vec![0; n];
        y.append(&mut vec![1; n]);
        println!("y: {:?}", y);
        y.shuffle(&mut rng);
        println!("Shuffled:   {:?}", y);
        let time1 = Instant::now();
        println!("time shuffle: {:?}", time1.saturating_duration_since(time0));

        let time0 = Instant::now();
        let mut x = Vec::with_capacity(2 * n);
        for _ in 0..2 * n {
            x.push(rng.gen_range(0..2));
        }
        println!("x: {:?}", x);
        let time1 = Instant::now();
        println!("time for gen_range: {:?}", time1.saturating_duration_since(time0));

        let time0 = Instant::now();
        let mut z = vec![0; 2 * n];
        z = z.iter().map(|&x| rng.gen_range(0..2)).collect();
        println!("z: {:?}", z);

        let time1 = Instant::now();
        println!("time for gen_range 2: {:?}", time1.saturating_duration_since(time0));
    }

    #[test]
    fn trycodeword() {
        let n = 1000000;
        let mut rng = rand::thread_rng();
        //let original_code_word: Vec<usize> = vec![0; n];

        let time0 = Instant::now();
        let mut indices = Vec::new();
        for i in 0..n {
            if rng.gen::<f64>() < 0.5 {
                indices.push(i);
            }
        }
        let nnz = indices.len();
        //println!("codeword indices: {:?} nnz: {}", indices, nnz);
        let sparse_vec = CsVec::new(n, indices, vec![1; nnz]);
        //println!("sparse vec : {:?}", sparse_vec);

        let time1 = Instant::now();
        println!("time for gen_range 2: {:?}", time1.saturating_duration_since(time0));

        let time0 = Instant::now();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let nnz = n / 2;
        indices.truncate(nnz);
        indices.sort();
        //println!("indices:   {:?}", indices);
        let sparse_vec = CsVec::new(n, indices, vec![1; nnz]);
        //println!("sparse vec : {:?}", sparse_vec);

        let time1 = Instant::now();
        println!("time shuffle: {:?}", time1.saturating_duration_since(time0));

        // keep codeword as a vec + indices

        let time0 = Instant::now();
        let mut codeword = vec![0; n];
        let mut indices = vec![0; n];
        let mut indices = Vec::new();
        for i in 0..n {
            if rng.gen::<f64>() < 0.5 {
                codeword[i] = 1;
                indices.push(i);
            }
        }
        let nnz = indices.len();
        //println!("codeword indices: {:?} nnz: {}", indices, nnz);
        let sparse_vec = CsVec::new(n, indices, vec![1; nnz]);
        //println!("sparse vec : {:?}", sparse_vec);

        let time1 = Instant::now();
        println!("time 3: {:?}", time1.saturating_duration_since(time0));
    }
}

fn main() {
    let mut time0;
    let mut time1;

    // let n = 12;
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("usage: cargo run <n>");
        return;
    }
    let n: usize = args[1].parse().expect("arg n must be an integer");
    let w_c = 3;
    let w_r = 6;
    let n = w_r * (n / w_r);
    let m = w_c * (n / w_r);
    let crossover_proba = 0.05;
    let seed = 10;

    println!("n={} m={} w_c={} w_r={} p={}", n, m, w_c, w_r, crossover_proba);

    let mut rng = rand::thread_rng();
    time0 = Instant::now();
    let mut codeword: Vec<usize> = vec![0; n];
    let mut indices = Vec::new();
    for i in 0..n {
        if rng.gen::<f64>() < 0.5 {
            codeword[i] = 1;
            indices.push(i);
        }
    }
    //let nnz = indices.len();
    //println!("codeword: {:?}", codeword);
    //println!("codeword indices: {:?} nnz: {}", indices, nnz);
    //let sparse_vec = CsVec::new(n, indices, vec![1; nnz]);
    //println!("sparse vec : {:?}", sparse_vec);
    time1 = Instant::now();
    println!("time to generate codeword: {:?}", time1.saturating_duration_since(time0));

    // let sparse_vec = CsVec::new(n, vec![0; 0], vec![1; 0]);

    //   let a = CsMat::new_csc((9,12),
    //                     vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
    //                     vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 7, 3, 5, 9, 11, 1, 6, 8, 10, 2, 4, 6, 9, 5, 7, 10, 11, 0, 1, 3, 8],
    //                     vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    //   println!("Hello!");

    time0 = Instant::now();
    let mut matrix = data_processing::make_matrix_regular_ldpc(w_c, w_r, n, seed);
    time1 = Instant::now();
    //println!("matrix: {:?}", matrix);
    //println!("matrix {:?}", matrix.to_dense());
    println!("time to generate matrix: {:?}", time1.saturating_duration_since(time0));
    //println!("matrix indice: {:?}", matrix);

    // time0 = Instant::now();
    // let mut syndrome = &matrix * &sparse_vec;
    // time1 = Instant::now();
    // println!("time to compute syndrome 1: {:?}", time1.saturating_duration_since(time0));
    // time0 = Instant::now();
    // //println!("syndrome: {:?}", syndrome);
    // syndrome = syndrome.map(|&x| x % 2); // should then extrac only the non zero value..

    // //println!("syndrome: {:?}", syndrome);
    // //println!("syndrome: {:?}", syndrome.indices());
    // //println!("received: {:?}", received);
    // time1 = Instant::now();
    // println!("time to compute syndrome +: {:?}", time1.saturating_duration_since(time0));

    time0 = Instant::now();
    let mut syndrome2 = data_processing::csmat_dot(&matrix, &codeword);
    time1 = Instant::now();
    println!("time to compute syndrome 2: {:?}", time1.saturating_duration_since(time0));

    //assert_eq!(syndrome2, syndrome.data());

    let nbTest = 10;
    let mut cnt_success = 0;
    for _ in 0..nbTest {
        time0 = Instant::now();
        let (mut received, mut post_proba) = data_processing::bsc_channel(n, &codeword, crossover_proba);
        //println!("recv: {:?}", received);
        let initial_errors: usize = received.iter().zip(codeword.iter()).map(|(&x, &y)| x ^ y).sum();
        println!("initial errors {}", initial_errors);
        time1 = Instant::now();
        println!("time to generate receive vector: {:?}", time1.saturating_duration_since(time0));

        time0 = Instant::now();
        let mut success: bool;
        let res = data_processing::message_passing(&mut matrix, &syndrome2, &post_proba, 60);
        success = res.0;
        received = res.1;
        //println!("decode: {:?}", received);
        //{
        // Some(value) => {
        //     println!("That is great but not quite!");
        //     //         assert_eq!(value.indices(), original_code_word)

        //
        // }
        // None => {
        //     println!("Sorry mate!")
        // }
        //}
        if success {
            cnt_success += 1;
            println!("success");
        } else {
            println!("failure");
        }

        let final_errors: usize = received.iter().zip(codeword.iter()).map(|(&x, &y)| x ^ y).sum();
        println!("final erros: {}", final_errors);

        time1 = Instant::now();
        println!("time to (not) decode: {:?}", time1.saturating_duration_since(time0));
    }
    println!("success: {}/{}", cnt_success, nbTest);
}
