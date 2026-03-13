use nonempty::NonEmpty;

pub fn contains_subword<T: Eq>(word: &[T], subword: &[T]) -> bool {
    if subword.len() > word.len() {
        return false;
    }
    word.windows(subword.len()).any(|window| window == subword)
}

pub fn contains_subword_v2<T: Eq>(_word: &NonEmpty<T>, _subword: &[T]) -> bool {
    todo!()
}
