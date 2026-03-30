## Python Topics Covered

### Core Python Basics

* Dynamic vs Static Typing
* Variables and Data Types

### Operators

* Arithmetic Operators
* Comparison Operators
* Logical Operators
* Bitwise Operators
* Identity Operators
* Membership Operators

### Package Management

* PIP (Package Installer for Python)

### Control Structures

* `if`
* `else`
* `elif`
* `for`
* `while`

### Exception Handling

* `try`
* `except`
* `finally`

### Functions

* Regular Functions
* Lambda Functions
* Recursive Functions
* Monkey Patching

### Data Structures

* Lists
* Tuples
* Sets
* Dictionaries

### Object-Oriented Programming (OOP)

* Classes
* Objects
* Constructors
* Class Methods
* Inheritance
* Encapsulation
* Abstraction

---

## Important Points to Remember in Python

* Python does not allow punctuation characters such as `@`, `$`, and `%` within identifiers.
* Python is a case-sensitive programming language.
* Class names usually start with an uppercase letter.
* Most other identifiers usually start with a lowercase letter.
* A single leading underscore indicates a private identifier.
* Two leading underscores indicate a strongly private identifier.
* If an identifier starts and ends with double underscores, it is considered a special language-defined name.

---

## Python Keywords

These are reserved words and cannot be used as variable names, constants, or identifiers:

```text
and, as, assert, break, class, continue, def, del, elif, else,
except, False, finally, for, from, global, if, import, in, is,
lambda, None, nonlocal, not, or, pass, raise, return, True, try,
while, with, yield
```

---

## Python Statements

### Line Endings

In Python, statements usually end with a new line. Python also allows the use of the line continuation character `\` when a statement needs to continue onto the next line.

### Quotation Marks

Python accepts:

* Single quotes: `' '`
* Double quotes: `" "`
* Triple quotes: `''' '''` or `""" """`

Triple quotes are used for multi-line strings.

### Multiple Statements on a Single Line

A semicolon `;` can be used to write multiple statements on the same line.

Example:

```python
x = 10; y = 20; print(x + y)
```

### Suites in Python

A group of statements that form a code block is called a suite.

Example:

```python
if x:
    print(y)
else:
    print(z)
```

---

## Python Variables

### Memory Addresses

Python manages memory automatically, unlike languages such as C++ where memory management is more manual.

* `id()` returns the memory identity of an object.
* Python uses dynamic memory allocation.
* When a variable is declared, Python automatically determines whether it is a string, integer, float, or another type, and assigns memory accordingly.

Example:

```python
str1 = "hello"
str2 = str1
```

In this case, both variables may point to the same memory location.

### Variable Concepts

* Creating Python variables
* Deleting Python variables
* Getting the type of a variable
* Casting variables
* Case sensitivity of variables
* Multiple assignment

---

## Python Variable Naming Conventions

Example:

```python
_con1 = 'first convention'
con2 = "second convention"
Age = "third convention"
test_salary = "fourth convention"
name22 = "fifth convention"
CONSTANT_VARIABLE = "Constant Variable"
```

### Naming Styles

#### Camel Case

The first word starts with a lowercase letter, and each following word starts with an uppercase letter.

Examples:

* `kmPerHour`
* `pricePerLitre`

#### Pascal Case

The first letter of each word is uppercase.

Examples:

* `KmPerHour`
* `PricePerLitre`

#### Snake Case

Words are separated using underscores.

Examples:

* `km_per_hour`
* `price_per_litre`

### Invalid Variable Names

```python
1counter = 100
$_count = 100
zara-salary = 100000
```

### Variable Scope

* Local Variables
* Global Variables
* Constants in Python

---

## Data Types in Python

### Main Data Types Supported in Python

* Numeric Data Types: `int`, `float`, `complex`
* String Data Type
* Sequence Data Types: `list`, `tuple`, `range`
* Binary Data Types: `bytes`, `bytearray`, `memoryview`
* Dictionary Data Type
* Set Data Type: `set`, `frozenset`
* Boolean Data Type
* None Type

---

## Lists and Tuples

A list in Python can contain multiple data types.

### List

* Written using square brackets `[]`
* Mutable, meaning it can be changed

### Tuple

* Written using parentheses `()`
* Immutable, meaning it cannot be changed
* Can be thought of as a read-only list

---

## Binary Data Types

Binary data types are used to represent data in binary form. They are often used for images, files, network packets, and raw data processing.

### `bytes`

Used to represent immutable byte sequences with values from `0` to `255`.

Example:

```python
bytes([23, 45, 65])
```

### `bytearray()`

Similar to `bytes()`, but mutable.

A `bytearray` can be created by:

* passing an iterable of integers
* encoding a string
* converting an existing `bytes` or `bytearray` object

Example:

```python
bytearray([65, 66, 67])
```

### `memoryview()`

Provides access to the memory of another binary object without copying its data.

This is useful for:

* large datasets
* better performance
* working directly with buffers

Example:

```python
data = bytes([65, 66, 67])
view = memoryview(data)
```

---

## Functions and Methods in Python

### Rules for Defining Function Names

Functions are defined using the `def` keyword.

Example:

```python
def my_function():
    pass
```

### Pass by Reference

In pass-by-reference, both the actual argument and the formal argument refer to the same memory location.

### Pass by Value

In pass-by-value, the value of the actual argument is copied into the formal argument.

### Types of Function Arguments in Python

* Positional Arguments
* Keyword Arguments
* Default Arguments
* Variable-Length Arguments (`*args`, `**kwargs`)

---

## Useful Links

### Python Roadmap

`https://roadmap.sh/python`

### Python Documentation

`https://docs.python.org/3/tutorial/`

### Python Data Types Reference

`https://www.tutorialspoint.com/python/python_data_types.htm`

### Google Docs Link

`https://docs.google.com/document/d/1BrIK5zeCY4AEIDkZ1QtylxEii73dg_Zt/edit`
