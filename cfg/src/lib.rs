pub trait ASTNode {}
pub trait Token {}
/*
==== GENERAL IDEA for CFG representation:

#[tokens]
enum Tokens {
    EOF,
    LET,
    IDENT(String),
    NUMERIC(f64),
    EQ,
    #[ignored]
    WHITESPACE
}

 |
 |
 |
\ /

struct EOF();
impl Token for EOF {};
struct LET();
impl Token for LET {};
struct IDENT(String);
impl Token for IDENT {};
struct NUMERIC(f64);
impl Token for NUMERIC {};
struct EQ();
impl Token for EQ {};
*/

/*
#[production]
enum Prod1 {
    Alt1(Prod1, NUMERIC),
    Alt2(NUMERIC)
}

 |
 |
 |
\ /

enum Prod1 {
    Alt1(Box<Prod1>, Box<NUMERIC>),
    Alt2(Box<NUMERIC>)
}
impl Token for Prod1 {}
impl ASTNode for Prod1 {}
*/
