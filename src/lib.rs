use std::collections::HashMap;
use ndarray::Array2;
use ndarray::arr1;
use ndarray::Array;
enum SuccessOrFail{
    Success(Vec<i8>),
    Fail(bool),
}

struct Data{
    neighbor_col: Vec<(usize,usize)>,
    neighbor_row: Vec<(usize,usize)>,
    q_0: f32,
    q_1: f32,
    r_0: f32,
    r_1: f32,
}

fn make_matrix(w_c: u32, w_r: u32, n: u32, seed: i8) -> (){
    if (n*w_c)%w_r != 0{
        panic!("number of col * weight of col must be divisible by weight of row");
    }
    let num_row: u32 = n*w_c/w_r;
    ()
}

pub fn run(m: usize,n: usize,matrix: Vec<Vec<i8>>, proba: Vec<(f32,f32)>, check: Vec<i8>,
           number_of_iter: usize) -> SuccessOrFail{
    let mut result : Vec<i8> = vec![0;n];
    let mut dict : HashMap<(usize, usize), Data> = HashMap::new();
    for i in 0..m{
        for j in 0..n{
            if matrix[i][j] == 1 {
                let mut neighborC : Vec<(usize, usize)> = Vec::new();
                let mut neighborR : Vec<(usize, usize)> = Vec::new();
                for k in 0..n{
                    if matrix[i][k] == 1 && k != j{ 
                        neighborR.push((i,k));
                    }
                }
                for k in 0..m{
                    if matrix[k][j] == 1 && k != i{
                        neighborC.push((k,j))
                    }
                }
                let (mut q_0,mut q_1) = proba[j];
                let (mut r_0,mut r_1) = (1.0,1.0);
                let data = Data{
                    neighbor_col: neighborC,
                    neighbor_row: neighborR,
                    q_0: q_0,
                    q_1: q_1,
                    r_0: r_0,
                    r_1: r_1,
                }; 
                dict.insert((i,j), data);
            }
        }
    }

    for i in 0..number_of_iter {
        horizontal_run(&mut dict);
        vertical_run(&mut dict, &proba, & mut result);
        match verification(&matrix,&result,&check){
            true => {
                return SuccessOrFail::Success(result);}
            false => {
                continue;
            }
        }
    }
    SuccessOrFail::Fail(false)
}

fn horizontal_run(dict: &mut HashMap<(usize, usize), Data>) 
                    -> &mut HashMap<(usize, usize), Data>{
    for (key, value) in dict{
        let mul : f32 = 1.0;
            for neighborR in dict[key].neighbor_row{
                mul *= dict.get(&(neighborR.0, neighborR.1)).unwrap().q_0 - dict.get(&(neighborR.0,neighborR.1)).unwrap().q_1;
            }
            let mut temp: &Data;
            match (*dict).get_mut(&key){
                Some(val) => temp = val,
                None => panic!("there is no value associated with key!!!"),
            }
            (*temp).r_0 =  (1.0+mul)/2.0;
            (*temp).r_0 =  (1.0-mul)/2.0;            
//            dict.insert(key, *temp);
    }
    dict
}

fn vertical_run<'a,'b>(dict: &'a mut HashMap<(usize, usize), Data>, proba: &Vec<(f32,f32)>, string: &'b mut Vec<i8>) 
                    -> (&'a mut HashMap<(usize, usize), Data>, &'b mut Vec<i8>){
    for (key, value) in dict{
        let mut q0 : f32 = proba[key.1].0;
        let mut q1 : f32 = proba[key.1].1;
        for neighborC in dict[key].neighbor_col{
            q0 *= dict.get_mut(&neighborC).unwrap().r_0;
            q1 *= dict.get_mut(&neighborC).unwrap().r_1;
        }
        let sum = q0+q1;
        q0 = q0/sum;
        q1 = q1/sum;
        let mut temp: &Data;
        match dict.get_mut(&key){
            Some(val) => temp = val,
            None => panic!("Cannot find value associated with key line 106 "),
        }
        (*temp).q_0 = q0;
        (*temp).q_1 = q1;
        q0 = q0*((*dict).get(&key).unwrap().q_0);
        q1 = q1*((*dict).get(&key).unwrap().q_1);
        if q1 >= q0{
            string[key.1] = 1; 
        }else{
            string[key.1] = 0;
        }
    }
    (dict, string)
}

fn verification(matrix: &Vec<Vec<i8>>, string: &Vec<i8>, check: &Vec<i8> ) -> bool{
    let matrix = Array2::from(*matrix);
    let string = Array::from(*string);
    let check = Array::from(*check);
    let res = matrix.dot(&string);
    for i in 0..res.len(){
        if res[i] != check[i]{return false;}
        else {continue;}
    }
    return true;
    
}
//#[cfg(test)]
//mod tests{
//    use super::*;
//    #[test]
//    fn horizontal_test(){
//        assert_eq!(vec![vec![0,1]], horizontal_run());
//    }
//}
