use lexer::{
    Stream,
    TokenStream,
    LexStatus
};

use std::{fmt, str, error};

#[derive(Debug, PartialEq, Clone)]
pub enum BinOp {
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    LT,
    LTE,
    GT,
    GTE,
    EQ
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    EOF,
    SPACE,
    DEF,
    LET,
    IF,
    ELSE,
    ASSIGNMENT,
    PAREN_OP,
    PAREN_CL,
    BLOCK_OP,
    BLOCK_CL,
    IDENT(String),
    BIN_OP(BinOp),
    NUM(f64)
}

#[derive(Debug)]
pub struct TokenzierError {
    text: String
}

impl TokenzierError {
    pub fn from(string: impl Into<String>) -> Self {
        Self {
            text: string.into()
        }
    }
}

impl error::Error for TokenzierError {}
impl fmt::Display for TokenzierError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let txt = format!("TokenizerError: {}", self.text);
        f.write_str(&txt)
    }
}

fn drop_comments(input: String) -> String {
    let input = str::replace(&input, "\r\n", "\n");
    input.split("\n").map(|line| {
        line.split("#").next().unwrap().to_string()
    }).fold(String::new(), |acc, x| {
        format!("{}\n{}", acc, x)
    })
}

pub fn mk_tokens(input: String) -> Result<Vec<Token>, TokenzierError> {
    let mut ts = TokenStream::<_, Token>::new();

    // Detecting EOF
    ts.rule(&|chunk, eof| {
        match (chunk.len() == 0, eof) {
            (true, true) => LexStatus::Token(Token::EOF),
            _            => LexStatus::Fail
        }
    });

    // Detecting whitespaces block
    ts.rule(&|chunk, eof| {
        match (chunk.starts_with(" "), chunk.ends_with(" "), eof) {
            (false, _, _)       => LexStatus::Fail,
            (true, true, true)  => LexStatus::Token(Token::SPACE),
            (true, true, false) => LexStatus::Request,
            (true, false, _)    => LexStatus::TokenWithDrop(Token::SPACE)
        }
    });

    // Special symbols
    ts.rule(&|chunk, _| {
        match chunk.as_str() {
            "="  => LexStatus::Token(Token::ASSIGNMENT),
            "("  => LexStatus::Token(Token::PAREN_OP),
            ")"  => LexStatus::Token(Token::PAREN_CL),
            "{"  => LexStatus::Token(Token::BLOCK_OP),
            "}"  => LexStatus::Token(Token::BLOCK_CL),
            "+"  => LexStatus::Token(Token::BIN_OP(BinOp::ADD)),
            "-"  => LexStatus::Token(Token::BIN_OP(BinOp::SUB)),
            "*"  => LexStatus::Token(Token::BIN_OP(BinOp::MUL)),
            "/"  => LexStatus::Token(Token::BIN_OP(BinOp::DIV)),
            "%"  => LexStatus::Token(Token::BIN_OP(BinOp::MOD)),
            _    => LexStatus::Fail
        }
    });

    // >, >=, <, <=, ==
    ts.rule(&|chunk, eof| {
        let chars = chunk
            .chars()
            .collect::<Vec<char>>();

        if chars.len() == 0 || eof{
            return LexStatus::Fail;
        }

        let first = *chars.first().unwrap();
        let last = *chars.last().unwrap();

        match (first, last, chars.len()) {
            ('<', '=', 2) => LexStatus::Token(Token::BIN_OP(BinOp::LTE)),
            ('<', _, 2)   => LexStatus::TokenWithDrop(Token::BIN_OP(BinOp::LT)),
            ('>', '=', 2) => LexStatus::Token(Token::BIN_OP(BinOp::GTE)),
            ('>', _, 2)   => LexStatus::TokenWithDrop(Token::BIN_OP(BinOp::GT)),
            ('=', '=', 2) => LexStatus::Token(Token::BIN_OP(BinOp::EQ)),
            ('=', _, 2)   => LexStatus::Fail,
            ('>', _, 1)   => LexStatus::Request,
            ('<', _, 1)   => LexStatus::Request,
            ('=', _, 1)   => LexStatus::Request,
            _             => LexStatus::Fail
        }
    });

    // Keywords
    ts.rule(&|chunk, _| {
        match chunk.as_str() {
            "d" | "de" | "def"          => LexStatus::Request,
            "def "                      => LexStatus::TokenWithDrop(Token::DEF),
            "l" | "le" | "let"          => LexStatus::Request,
            "let "                      => LexStatus::TokenWithDrop(Token::LET),
            "i" | "if"                  => LexStatus::Request,
            "if " | "if("               => LexStatus::TokenWithDrop(Token::IF),
            "e" | "el" | "els" | "else" => LexStatus::Request,
            "else " | "else{"           => LexStatus::TokenWithDrop(Token::ELSE),
            _                           => LexStatus::Fail
        }
    });

    // Numeric values
    ts.rule(&|chunk, eof| {
        let chars = chunk
            .chars()
            .collect::<Vec<char>>();

        if chars.len() == 0 {
            return LexStatus::Fail;
        }

        let dots_cnt = chars
            .iter()
            .filter(|chr| **chr == '.')
            .count();

        if dots_cnt > 1 {
            return LexStatus::Fail;
        }

        let first = *chars.first().unwrap();
        let last = *chars.last().unwrap();

        if last == '.' && eof {
            return LexStatus::Fail;
        }

        match (first.is_numeric(), last.is_numeric() || last == '.', eof) {
            (false, _, _)       => LexStatus::Fail,
            (true, true, true)  => {
                match chunk.parse::<f64>() {
                    Ok(num) => LexStatus::Token(Token::NUM(num)),
                    _       => LexStatus::Fail
                }
            },
            (true, true, false) => LexStatus::Request,
            (true, false, _)    => {
                let cand = &chunk[..chunk.len()-1].to_string();
                match cand.parse::<f64>() {
                    Ok(num) => LexStatus::TokenWithDrop(Token::NUM(num)),
                    _ => LexStatus::Fail
                }
            }
        }
    });

    // Identifiers
    ts.rule(&|chunk, eof| {
        let chars = chunk.chars().collect::<Vec<char>>();
        if chars.len() == 0 {
            return LexStatus::Fail;
        }

        let first = *chars.first().unwrap();
        let last = *chars.last().unwrap();

        match (first.is_ascii_alphabetic(), last.is_ascii_alphanumeric(), eof) {
            (false, _, _) =>
                LexStatus::Fail,
            (_, false, _) =>
                LexStatus::TokenWithDrop(Token::IDENT(String::from(&chunk[..chunk.len()-1]))),
            (_, _, true) =>
                LexStatus::Token(Token::IDENT(String::from(chunk))),
            _          =>
                LexStatus::Request
        }
    });

    // Replace newlines with whitespaces
    let input = drop_comments(input)
        .chars()
        .map(|chr| {
            match chr {
                '\n' | '\r' | '\t' => ' ',
                x                  => x
            }
        }).collect::<String>();
    ts.string(input);

    let mut tokens = Vec::new();
    loop {
        let next = ts.next();
        match next {
            None               => {
                let err = ts.error().unwrap_or("Unrecognized token.".to_string());
                return Err(TokenzierError::from(err));
            },
            // EOF means end has been reached successfully
            Some(Token::EOF)   => break,
            // Ignore whitespaces
            Some(Token::SPACE) => {},
            // Otherwise add token to the output list
            Some(token)        => tokens.push(token)
        }
    }

    Ok(tokens)
}
