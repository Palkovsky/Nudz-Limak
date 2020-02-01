mod token;

use token::*;
use std::io::{self, Read};

fn main() -> io::Result<()> {
    // Read STDIN
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;

    // Generate tokens
    let tokens = mk_tokens(buffer);
    println!("{:?}", tokens);

    Ok(())
}
