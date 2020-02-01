mod stream;
pub use stream::*;

#[derive(Clone)]
pub struct CharStream {
    chars: Vec<char>,
    stack: Vec<char>
}

impl CharStream {
    pub fn from(string: impl Into<String>) -> CharStream {
        let as_str = string.into();
        CharStream {
            chars: as_str.chars().collect(),
            stack: Vec::with_capacity(as_str.len())
        }
    }

    pub fn consumed(&self) -> String {
        self.stack.iter().fold(String::new(), |acc, x| format!("{}{}", acc, x))
    }

    pub fn left(&self) -> String {
        self.chars.iter().fold(String::new(), |acc, x| format!("{}{}", acc, x))
    }
}

impl Stream<char> for CharStream {
    fn next(&mut self) -> Option<char> {
        let x = *self.chars.first()?;
        self.stack.push(x);
        self.chars.remove(0);
        Some(x)
    }

    fn revert(&mut self) -> Option<char> {
        let x = self.stack.pop()?;
        self.chars.insert(0, x);
        Some(x)
    }
}

#[test]
fn char_stream_test() {
    let mut stream = CharStream::from("xDD");
    assert_eq!("", stream.consumed());
    assert_eq!("xDD", stream.left());

    assert_eq!(None, stream.revert());

    assert_eq!(Some('x'), stream.next());
    assert_eq!("x", stream.consumed());
    assert_eq!("DD", stream.left());

    assert_eq!(Some('D'), stream.next());
    assert_eq!("xD", stream.consumed());
    assert_eq!("D", stream.left());

    assert_eq!(Some('D'), stream.next());
    assert_eq!("xDD", stream.consumed());
    assert_eq!("", stream.left());

    assert_eq!(None, stream.next());
    assert_eq!(Some('D'), stream.revert());
    assert_eq!(Some('D'), stream.revert());
    assert_eq!(Some('x'), stream.revert());

    assert_eq!(None, stream.revert());
}

#[derive(Clone, Copy)]
pub enum LexStatus<T> {
    Request,
    Fail,
    Token(T),
    TokenWithDrop(T)
}

#[derive(Clone)]
pub struct TokenStream<'r, S: Stream<char>, T: Clone> {
    char_stream: S,
    rules: Vec<&'r dyn Fn(String, bool) -> LexStatus<T>>,
    stack: Vec<(T, usize)>,
    error: Option<String>
}

// Constructor for the concrete CharStream
impl<'r, T: Clone> TokenStream<'r, CharStream, T> {
    pub fn from_string(string: impl Into<String>) -> Self {
        Self::from_char_stream(CharStream::from(string))
    }

    pub fn new() -> Self {
        Self::from_char_stream(CharStream::from(String::new()))
    }

    pub fn string(&mut self, stream: impl Into<String>) -> &mut Self {
        self.char_stream = CharStream::from(stream);
        self.error = None;
        self
    }
}

impl<'r, T: Clone, S: Stream<char>> TokenStream<'r, S, T> {
    pub fn from_char_stream(char_stream: S) -> Self {
        Self {
            char_stream: char_stream,
            rules: Vec::new(),
            stack: Vec::new(),
            error: None
        }
    }

    pub fn rule(&mut self, func: &'r dyn Fn(String, bool) -> LexStatus<T>) -> &mut Self {
        self.rules.push(func);
        self
    }

    pub fn stream(&mut self, stream: S) -> &mut Self {
        self.char_stream = stream;
        self.error = None;
        self
    }

    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }
}

impl<'r, T: Clone, S: Stream<char>> Stream<T> for TokenStream<'r, S, T> {
    fn next(&mut self) -> Option<T> {
        self.error = None;
        for i in 0..self.rules.len() {
            let mut finished = false;
            let mut chunk_size = 1;

            while !finished {
                let chunk = (0..chunk_size)
                    .map(|_| self.char_stream.next())
                    .fold(String::new(), |acc, chr| {
                        let joined = chr.map(|x| format!("{}{}", acc, x));
                        if let Some(y) = joined {
                            y
                        } else {
                            acc
                        }
                    });

                let eof = self.char_stream.is_finished();
                let func = self.rules.get(i).unwrap();

                let revert_chars = |stream: &mut S| for _ in 0..chunk_size {
                    stream.revert();
                };

                match func(chunk, eof) {
                    // Lexer requested one more character
                    LexStatus::Request => {
                        revert_chars(&mut self.char_stream);
                        chunk_size += 1;
                    },
                    // Rule didn't recognize token
                    LexStatus::Fail => {
                        revert_chars(&mut self.char_stream);
                        finished = true;
                    },
                    // Rule recognized token
                    LexStatus::Token(token) => {
                        self.stack.push((token.clone(), chunk_size));
                        return Some(token);
                    },
                    // Rule recognized token, but consumed one extra character
                    LexStatus::TokenWithDrop(token) => {
                        self.stack.push((token.clone(), chunk_size-1));
                        self.char_stream.revert();
                        return Some(token);
                    }
                }
            }
        }

        // Add error message
        let err = self.char_stream.peek(5)
            .into_iter()
            .fold(String::new(), |acc, x| format!("{}{}", acc, x));
        self.error = Some(format!("Invalid token '{}'", err));
        None
    }

    fn revert(&mut self) -> Option<T> {
        let (token, len) = self.stack.pop()?;
        self.error = None;
        for _ in 0..len {
            self.char_stream.revert();
        }
        Some(token)
    }
}

#[derive(Debug, PartialEq, Clone)]
enum BinOp {
    PLUS, MINUS, MODULO
}

#[derive(Debug, PartialEq, Clone)]
enum TestToken {
    EOF,
    LET,
    IDENT(String),
    NUMERIC(f64),
    EQ,
    BIN_OP(BinOp),
    WHITESPACE,
}

#[test]
fn token_stream_error_test() {
    let mut ts = TokenStream::from_string("1+1=1xd2+2=2");

    ts.rule(&|chunk, eof| {
        match (chunk.len() == 0, eof) {
            (true, true) => LexStatus::Token(TestToken::EOF),
            _            => LexStatus::Fail
        }
    }).rule(&|chunk, _| {
        match chunk.as_str() {
            "=" => LexStatus::Token(TestToken::EQ),
            "+" => LexStatus::Token(TestToken::BIN_OP(BinOp::PLUS)),
            "-" => LexStatus::Token(TestToken::BIN_OP(BinOp::MINUS)),
            "%" => LexStatus::Token(TestToken::BIN_OP(BinOp::MODULO)),
            _   => LexStatus::Fail
        }
    }).rule(&|chunk, eof| {
        let chars = chunk.chars().collect::<Vec<char>>();
        if chars.len() == 0 {
            return LexStatus::Fail;
        }

        let first = *chars.first().unwrap();
        let last = *chars.last().unwrap();

        match (first.is_numeric(), last.is_numeric() || last == '.', eof) {
            (false, _, _)       => LexStatus::Fail,
            (true, true, true)  => {
                match chunk.parse::<f64>() {
                    Ok(num) => LexStatus::Token(TestToken::NUMERIC(num)),
                    _       => LexStatus::Fail
                }
            },
            (true, true, false) => LexStatus::Request,
            (true, false, _)    => {
                let cand = &chunk[..chunk.len()-1].to_string();
                match cand.parse::<f64>() {
                    Ok(num) => LexStatus::TokenWithDrop(TestToken::NUMERIC(num)),
                    _ => LexStatus::Fail
                }
            }
        }
    });

    assert_eq!(Some(TestToken::NUMERIC(1f64)),       ts.next());
    assert_eq!(Some(TestToken::BIN_OP(BinOp::PLUS)), ts.next());
    assert_eq!(Some(TestToken::NUMERIC(1f64)),       ts.next());
    assert_eq!(Some(TestToken::EQ),                  ts.next());
    assert_eq!(Some(TestToken::NUMERIC(1f64)),       ts.next());
    assert!(ts.error().is_none());
    assert_eq!(None,                                 ts.next());
    assert!(ts.error().is_some());
    assert_eq!(Some(TestToken::NUMERIC(1f64)),       ts.revert());
    assert!(ts.error().is_none());
    assert_eq!(Some(TestToken::NUMERIC(1f64)),       ts.next());
    assert_eq!(None,                                 ts.next());
    assert!(ts.error().is_some());
    assert_eq!(None,                                 ts.next());
    assert!(ts.error().is_some());
    assert_eq!(None,                                 ts.next());
    assert!(ts.error().is_some());
}

#[test]
fn token_stream_test() {
    let mut ts = TokenStream::new();
    ts.rule(&|chunk, eof| {
        match (chunk.len() == 0, eof) {
            (true, true) => LexStatus::Token(TestToken::EOF),
            _            => LexStatus::Fail
        }
    }).rule(&|chunk, eof| {
        match (chunk.starts_with(" "), chunk.ends_with(" "), eof) {
            (false, _, _)       => LexStatus::Fail,
            (true, true, true)  => LexStatus::Token(TestToken::WHITESPACE),
            (true, true, false) => LexStatus::Request,
            (true, false, _)    => LexStatus::TokenWithDrop(TestToken::WHITESPACE)
        }
    }).rule(&|chunk, _| {
        if chunk == "=" {
            LexStatus::Token(TestToken::EQ)
        } else {
            LexStatus::Fail
        }
    }).rule(&|chunk, _| {
        match chunk.as_str() {
            "l"    => LexStatus::Request,
            "le"   => LexStatus::Request,
            "let"  => LexStatus::Request,
            "let " => LexStatus::TokenWithDrop(TestToken::LET),
            _      => LexStatus::Fail
        }
    }).rule(&|chunk, _| {
        let chars = chunk.chars().collect::<Vec<char>>();
        if chars.len() == 0 {
            return LexStatus::Fail;
        }

        let first = *chars.first().unwrap();
        let last = *chars.last().unwrap();

        match (first.is_numeric(), last.is_numeric() || last == '.') {
            (false, _) => LexStatus::Fail,
            (true, true) => LexStatus::Request,
            (true, false) => {
                let cand = &chunk[..chunk.len()-1].to_string();
                match cand.parse::<f64>() {
                    Ok(num) => LexStatus::TokenWithDrop(TestToken::NUMERIC(num)),
                    _ => LexStatus::Fail
                }
            }
        }
    }).rule(&|chunk, _| {
        let chars = chunk.chars().collect::<Vec<char>>();
        if chars.len() == 0 {
            return LexStatus::Fail;
        }

        let first = *chars.first().unwrap();
        let last = *chars.last().unwrap();

        match (first.is_ascii_alphabetic(), last.is_ascii_alphanumeric()) {
            (false, _) => LexStatus::Fail,
            (_, false) => LexStatus::TokenWithDrop(TestToken::IDENT(String::from(&chunk[..chunk.len()-1]))),
            _          => LexStatus::Request
        }
    });

    ts.string(" let  letXDD21 = 21 12xd ");
    assert_eq!(Some(TestToken::WHITESPACE),                   ts.next());
    assert_eq!(Some(TestToken::LET),                          ts.next());
    assert_eq!(Some(TestToken::WHITESPACE),                   ts.next());
    assert_eq!(Some(TestToken::IDENT("letXDD21".to_owned())), ts.next());
    assert_eq!(Some(TestToken::WHITESPACE),                   ts.next());
    assert_eq!(Some(TestToken::EQ),                           ts.next());
    assert_eq!(Some(TestToken::WHITESPACE),                   ts.next());
    assert_eq!(Some(TestToken::NUMERIC(21f64)),               ts.next());
    assert_eq!(Some(TestToken::WHITESPACE),                   ts.next());
    assert_eq!(Some(TestToken::NUMERIC(12f64)),               ts.next());
    assert_eq!(Some(TestToken::NUMERIC(12f64)),               ts.revert());
    assert_eq!(Some(TestToken::NUMERIC(12f64)),               ts.next());
    assert_eq!(Some(TestToken::IDENT("xd".to_owned())),       ts.next());
    assert_eq!(Some(TestToken::WHITESPACE),                   ts.next());
    assert_eq!(Some(TestToken::EOF),                          ts.next());
    assert_eq!(Some(TestToken::EOF),                          ts.next());
}
