"""Readable HTML evidence; source instructions remain data and links are not fetched.

Raw extraction remains in the caller's log. Offsets after rendering refer to
this labelled representation, not original HTML bytes.
"""

import hashlib
import re
from html.parser import HTMLParser


class _DocumentText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.stack = [], []
        self.math, self.tex = None, None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = bool(self.stack and self.stack[-1][1]) or tag in {'head', 'script', 'style', 'svg'}
        hidden |= 'katex-html' in attrs.get('class', '').split()
        if tag not in {'br', 'hr', 'img', 'meta', 'link', 'input', 'wbr', 'source'}:
            self.stack.append((tag, hidden, attrs))
        if hidden:
            return
        if tag == 'math':
            self.math, self.tex = [], []
        elif self.math is None:
            if re.fullmatch('h[1-6]', tag):
                self.parts.append('\n\n' + '#' * int(tag[1]) + ' ')
            elif tag == 'pre':
                self.parts.append('\n\n```\n')
            elif tag == 'code' and not any(t == 'pre' for t, _, _ in self.stack):
                self.parts.append('`')
            elif tag in {'p', 'div', 'section', 'article', 'ul', 'ol', 'table', 'blockquote'}:
                self.parts.append('\n\n')
            elif tag in {'br', 'tr'}:
                self.parts.append('\n')
            elif tag == 'li':
                self.parts.append('\n- ')
            elif tag in {'td', 'th'}:
                self.parts.append('| ')
            elif tag == 'img':
                self.parts.append(attrs.get('alt', ''))

    def handle_endtag(self, tag):
        index = next((i for i in range(len(self.stack)-1, -1, -1) if self.stack[i][0] == tag), None)
        if index is None:
            return
        _, hidden, attrs = self.stack[index]
        del self.stack[index:]
        if hidden:
            return
        if tag == 'math' and self.math is not None:
            self.parts.append('$' + ''.join(self.tex or self.math).strip() + '$')
            self.math, self.tex = None, None
        elif self.math is None:
            if tag == 'pre':
                self.parts.append('\n```\n\n')
            elif tag == 'code' and not any(t == 'pre' for t, _, _ in self.stack):
                self.parts.append('`')
            elif tag == 'a' and attrs.get('href'):
                self.parts.append(' <' + attrs['href'] + '>')
            elif tag in {'td', 'th'}:
                self.parts.append(' ')
            elif tag == 'tr':
                self.parts.append('|\n')
            elif tag in {'p', 'div', 'section', 'article', 'li', 'table', 'blockquote'} or re.fullmatch('h[1-6]', tag):
                self.parts.append('\n\n')

    def handle_data(self, data):
        if self.stack and self.stack[-1][1]:
            return
        if self.math is not None:
            if any(t == 'annotation' and a.get('encoding') == 'application/x-tex' for t, _, a in self.stack):
                self.tex.append(data)
            else:
                self.math.append(data)
        else:
            self.parts.append(data)


def readable_source(text):
    """Render HTML documents/fragments, not Markdown code examples."""
    if not re.match(r'\s*(?:<!doctype\s+html\b|<(?:html|body|article|div|p|h[1-6])(?:\s|>))', text, re.I):
        return text, None
    parser = _DocumentText()
    parser.feed(text)
    parser.close()
    rendered = ''.join(parser.parts).strip()
    if not rendered:
        return text, None
    return rendered, {'format': 'html-readable-v1', 'raw_chars': len(text), 'readable_chars': len(rendered),
                      'raw_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'offset_basis': 'rendered text; original extraction retained separately'}
