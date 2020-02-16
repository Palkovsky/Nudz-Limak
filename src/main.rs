mod token;
mod ast;
mod utils;
mod types;
mod allocator;
mod codegen;

use token::mk_tokens;
use ast::mk_ast;
use std::io::{self, Read};

fn main() -> () {
    let print_header = |text: &str| {
        print!("{:=<1$}", "", 30);
        print!("  {}  ", text);
        println!("{:=<1$}", "", 30);
    };

    // Read STDIN
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)
        .unwrap_or_else(|err| panic!("Unable to read stdin: {}", err));

    // Generate tokens
    print_header("LEXER");
    let mut tokens = mk_tokens(buffer).unwrap();
    println!("{:?}", tokens);

    // Parse token stream
    print_header("AST");
    let ast = mk_ast(&mut tokens);
    println!("{:#?}", ast);

    // Generate LLVM IR
    if let Ok(root) = ast {
        let (disasm, run) = codegen::gencode(&root);
        print_header("IR");
        println!("{}", disasm);

        print_header("EXECUTION");
        println!("Status: {}", run());
    }
    std::process::exit(0)
}
