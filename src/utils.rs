use nonempty::NonEmpty;

pub fn contains_subword<T: Eq>(word: &[T], subword: &[T]) -> bool {
    if subword.len() > word.len() {
        return false;
    }
    word.windows(subword.len()).any(|window| window == subword)
}

pub fn contains_subword_v2<T: Eq>(word: &NonEmpty<T>, subword: &[T]) -> bool {
    if subword.is_empty() {
        return true;
    }
    if subword[0] == *word.first() {
        let first_match = word.tail().starts_with(&subword[1..]);
        first_match || contains_subword(word.tail(), subword)
    } else {
        contains_subword(word.tail(), subword)
    }
}
