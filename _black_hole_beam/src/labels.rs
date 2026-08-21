//! Animal label formatting: shortening type names and extracting warp
//! boundary labels.

pub(crate) fn short_type_name<T: ?Sized>() -> String {
    animal_label_key(core::any::type_name::<T>())
}

pub(crate) fn animal_label_key(label: &str) -> String {
    short_type_label(label)
}

/// Extracts the boundary animal label from a warp node label of the form
/// `Warp<WarpAnimal, BoundaryAnimal>`.
///
/// Animal labels may carry generic arguments with nested angle brackets and
/// commas, so the split happens at the first top-level comma.
pub(crate) fn warp_boundary_label(label: &str) -> Option<String> {
    let inner = label.strip_prefix("Warp<")?.strip_suffix('>')?;
    let mut depth = 0i32;
    for (index, char) in inner.char_indices() {
        match char {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return Some(animal_label_key(&inner[index + 1..])),
            _ => {}
        }
    }
    None
}

fn short_type_label(label: &str) -> String {
    let mut shortened = String::with_capacity(label.len());
    let mut token = String::new();

    for ch in label.chars() {
        let token_char = ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':' | '\'');
        if token_char {
            token.push(ch);
            continue;
        }

        push_shortened_type_token(&mut shortened, &mut token);
        shortened.push(ch);
    }

    push_shortened_type_token(&mut shortened, &mut token);
    shortened.trim().to_string()
}

fn push_shortened_type_token(shortened: &mut String, token: &mut String) {
    if token.is_empty() {
        return;
    }

    let short = token.rsplit("::").next().unwrap_or(token.as_str());
    shortened.push_str(short);
    token.clear();
}
