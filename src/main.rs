pub mod token;
use token::mk_tokens;

pub mod ast;
use ast::mk_ast;

mod codegen;

use std::io::{self, Read};

fn main() -> io::Result<()> {
    // Read STDIN
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;

    // Generate tokens
    let mut tokens = mk_tokens(buffer).unwrap();
    println!("{:?}", tokens);

    let ast = mk_ast(&mut tokens);
    println!("{:#?}", ast);


    Ok(())
}
