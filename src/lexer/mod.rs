pub trait Stream<T> {
    fn next(&mut self) -> Option<T>;
    fn revert(&mut self) -> Option<T>;

    fn is_finished(&mut self) -> bool {
        let item = self.next();
        if item.is_some() {
            self.revert();
            false
        } else {
            true
        }
    }
}

struct CharStream {
    chars: Vec<char>,
    stack: Vec<char>
}

impl CharStream {
    fn from(string: impl Into<String>) -> CharStream {
        let as_str = string.into();
        CharStream {
            chars: as_str.chars().collect(),
            stack: Vec::with_capacity(as_str.len())
        }
    }

    fn consumed(&self) -> String {
        self.stack.iter().fold(String::new(), |acc, x| format!("{}{}", acc, x))
    }

    fn left(&self) -> String {
        self.chars.iter().fold(String::new(), |acc, x| format!("{}{}", acc, x))
    }
}

impl Stream<char> for CharStream {
    fn next(&mut self) -> Option<char> {
        let first = self.chars.first().map(|x| *x);
        if let Some(x) = first {
            self.stack.push(x);
            self.chars.remove(0);
        }
        first
    }

    fn revert(&mut self) -> Option<char> {
        let last = self.stack.pop();
        if let Some(x) = last {
            self.chars.insert(0, x);
        }
        last
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
pub enum RuleStatus<T> {
    Request,
    Fail,
    Token(T),
    TokenWithDrop(T)
}

pub struct TokenStream<'r, S: Stream<char>, T: Clone> {
    char_stream: S,
    rules: Vec<&'r dyn Fn(String, bool) -> RuleStatus<T>>,
    stack: Vec<(T, usize)>
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
        self
    }
}

impl<'r, T: Clone, S: Stream<char>> TokenStream<'r, S, T> {
    pub fn from_char_stream(char_stream: S) -> Self {
        Self {
            char_stream: char_stream,
            rules: Vec::new(),
            stack: Vec::new()
        }
    }

    pub fn rule(&mut self, func: &'r dyn Fn(String, bool) -> RuleStatus<T>) -> &mut Self {
        self.rules.push(func);
        self
    }

    pub fn stream(&mut self, stream: S) -> &mut Self {
        self.char_stream = stream;
        self
    }
}

impl<'r, T: Clone, S: Stream<char>> Stream<T> for TokenStream<'r, S, T> {
    fn next(&mut self) -> Option<T> {
        for i in 0..self.rules.len() {
            let mut finished = false;
            let mut chunk_size = 1;

            while !finished {
                let chunk = (0..chunk_size)
                    .map(|_| self.char_stream.next())
                    .fold(String::new(), |acc, chr| {
                        if let Some(y) = chr.map(|x| format!("{}{}", acc, x)) {
                            y
                        } else {
                            acc
                        }
                    });
                let eof = self.char_stream.is_finished();
                let func = self.rules.get(i).unwrap();
                let revert_chars = |stream: &mut S| (0..chunk_size).for_each(|_| {
                    stream.revert();
                });

                match func(chunk, eof) {
                    // Lexer requested one more character
                    RuleStatus::Request => {
                        revert_chars(&mut self.char_stream);
                        chunk_size += 1;
                    },
                    // Rule didn't recognize token
                    RuleStatus::Fail => {
                        revert_chars(&mut self.char_stream);
                        finished = true;
                    },
                    // Rule recognized token
                    RuleStatus::Token(token) => {
                        self.stack.push((token.clone(), chunk_size));
                        return Some(token);
                    },
                    // Rule recognized token, but consumed one extra character
                    RuleStatus::TokenWithDrop(token) => {
                        self.stack.push((token.clone(), chunk_size-1));
                        self.char_stream.revert();
                        return Some(token);
                    }
                }
            }
        }
        None
    }

    fn revert(&mut self) -> Option<T> {
        let last = self.stack.pop();
        if let Some((token, len)) = last {
            (0..len).for_each(|_| {
                self.char_stream.revert();
            });
            Some(token)
        } else {
            None
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
enum TestToken {
    EOF,
    LET,
    IDENT(String),
    NUMERIC(f64),
    EQ,
    WHITESPACE,
}

#[test]
fn token_stream_test() {
    let mut ts = TokenStream::new();
    ts.rule(&|chunk, eof| {
        match (chunk.len() == 0, eof) {
            (true, true) => RuleStatus::Token(TestToken::EOF),
            _            => RuleStatus::Fail
        }
    }).rule(&|chunk, eof| {
        match (chunk.starts_with(" "), chunk.ends_with(" "), eof) {
            (false, _, _)       => RuleStatus::Fail,
            (true, true, true)  => RuleStatus::Token(TestToken::WHITESPACE),
            (true, true, false) => RuleStatus::Request,
            (true, false, _)    => RuleStatus::TokenWithDrop(TestToken::WHITESPACE)
        }
    }).rule(&|chunk, _| {
        if chunk == "=" {
            RuleStatus::Token(TestToken::EQ)
        } else {
            RuleStatus::Fail
        }
    }).rule(&|chunk, _| {
        match chunk.as_str() {
            "l"    => RuleStatus::Request,
            "le"   => RuleStatus::Request,
            "let"  => RuleStatus::Request,
            "let " => RuleStatus::TokenWithDrop(TestToken::LET),
            _      => RuleStatus::Fail
        }
    }).rule(&|chunk, _| {
        let chars = chunk.chars().collect::<Vec<char>>();
        if chars.len() == 0 {
            return RuleStatus::Fail;
        }

        let first = *chars.first().unwrap();
        let last = *chars.last().unwrap();

        match (first.is_numeric(), last.is_numeric() || last == '.') {
            (false, _) => RuleStatus::Fail,
            (true, true) => RuleStatus::Request,
            (true, false) => {
                let cand = &chunk[..chunk.len()-1].to_string();
                match cand.parse::<f64>() {
                    Ok(num) => RuleStatus::TokenWithDrop(TestToken::NUMERIC(num)),
                    _ => RuleStatus::Fail
                }
            }
        }

    }).rule(&|chunk, _| {
        let chars = chunk.chars().collect::<Vec<char>>();
        if chars.len() == 0 {
            return RuleStatus::Fail;
        }

        let first = chars.first().unwrap();
        let last = chars.last().unwrap();

        if !first.is_ascii_alphabetic() {
            return RuleStatus::Fail;
        }

        if !last.is_ascii_alphanumeric() {
            return RuleStatus::TokenWithDrop(TestToken::IDENT(String::from(&chunk[..chunk.len()-1])));
        }

        return RuleStatus::Request;
    });

    ts.string(" let  letXDD21 = 21 12xd ");
    assert_eq!(Some(TestToken::WHITESPACE), ts.next());
    assert_eq!(Some(TestToken::LET), ts.next());
    assert_eq!(Some(TestToken::WHITESPACE), ts.next());
    assert_eq!(Some(TestToken::IDENT("letXDD21".to_owned())), ts.next());
    assert_eq!(Some(TestToken::WHITESPACE), ts.next());
    assert_eq!(Some(TestToken::EQ), ts.next());
    assert_eq!(Some(TestToken::WHITESPACE), ts.next());
    assert_eq!(Some(TestToken::NUMERIC(21f64)), ts.next());
    assert_eq!(Some(TestToken::WHITESPACE), ts.next());
    assert_eq!(Some(TestToken::NUMERIC(12f64)), ts.next());
    assert_eq!(Some(TestToken::NUMERIC(12f64)), ts.revert());
    assert_eq!(Some(TestToken::NUMERIC(12f64)), ts.next());
    assert_eq!(Some(TestToken::IDENT("xd".to_owned())), ts.next());
    assert_eq!(Some(TestToken::WHITESPACE), ts.next());
    assert_eq!(Some(TestToken::EOF), ts.next());
    assert_eq!(Some(TestToken::EOF), ts.next());
}
