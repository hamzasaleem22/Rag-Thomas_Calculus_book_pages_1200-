PAGE_RANGES = {
    "front_matter": (1, 5),
    "table_of_contents": (6, 11),
    "preface": (12, 21),
    "answer_key": (1100, 1262),
}

CHAPTER_START_PAGE = 22


def get_book_page(pdf_page: int) -> int:
    if pdf_page < CHAPTER_START_PAGE:
        return pdf_page
    return pdf_page - 21


def classify_page(page_num: int) -> str:
    for section, (start, end) in PAGE_RANGES.items():
        if start <= page_num <= end:
            return section
    return "chapter_content"


def get_chapter_number(page_num: int) -> int:
    if page_num < CHAPTER_START_PAGE:
        return 0
    chapter_map = {
        22: 1, 53: 1,
        54: 2, 94: 2,
        95: 3, 161: 3,
        162: 4, 230: 4,
        231: 5, 309: 5,
        310: 6, 381: 6,
        382: 7, 439: 7,
        440: 8, 505: 8,
        506: 9, 558: 9,
        559: 10, 649: 10,
        650: 11, 705: 11,
        706: 12, 755: 12,
        756: 13, 803: 13,
        804: 14, 887: 14,
        888: 15, 961: 15,
        962: 16, 1053: 16,
    }
    for start, ch_num in sorted(chapter_map.items(), reverse=True):
        if page_num >= start:
            return ch_num
    return 0


TOC_CHAPTER_NAMES = {
    1: "Functions",
    2: "Limits and Continuity",
    3: "Derivatives",
    4: "Applications of Derivatives",
    5: "Integrals",
    6: "Applications of Definite Integrals",
    7: "Integrals and Transcendental Functions",
    8: "Techniques of Integration",
    9: "First-Order Differential Equations",
    10: "Infinite Sequences and Series",
    11: "Parametric Equations and Polar Coordinates",
    12: "Vectors and the Geometry of Space",
    13: "Vector-Valued Functions and Motion in Space",
    14: "Partial Derivatives",
    15: "Multiple Integrals",
    16: "Integrals and Vector Fields",
}


def get_chapter_name(page_num: int) -> str:
    ch_num = get_chapter_number(page_num)
    name = TOC_CHAPTER_NAMES.get(ch_num, "")
    return f"Chapter {ch_num} {name}" if name and ch_num > 0 else ""
