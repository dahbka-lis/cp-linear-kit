# CP 2024: LinearKit - linear algebra for C++

`LinearKit` - это библиотека для работы с матрицами на языке C++ стандарта 20, предоставляющая матричные алгоритмы и разложения, такие как:

- Отражения Хаусхолдера и повороты Гивенса.
- QR разложение.
- QR алгоритм для симметричных матриц через форму Хессенберга.
- QR алгоритм для бидиагональных матриц со сдвигами Уилкинсона.
- Сингулярное разложение матрицы.

## 🛠️ Работа с библиотекой

---

Библиотека обернута в пространство имён `LinearKit` и содержит в себе:

- Шаблонные матричные типы `Matrix<T>`, `MatrixView<T>` и `ConstMatrixView<T>`, причём тип `T` может быть только вещественным или `std::complex<T>` на основе вещественного.
- Пространство `Algorithm` с имплементацией алгоритмов, перечисленных выше.
- Пространство `MatrixUtils` с полезными матричными концептами и функциями для работы алгоритмов.
- Пространство `Utils` для всего полезного, что не связано с матрицами.

Пример работы с библиотекой:

```cpp
using namespace std;
using namespace LinearKit;

Matrix<float> m = {{1, 7, 4}, {9, 2, 5}, {-6, 1, 8}};
MatrixView<float> view = m.GetSubmatrix({1, -1}, {1, -1}); // [[2, 5], [1, 8]]

auto [Q, R] = Algorithm::HouseholderQR(view);
auto I = Q * Matrix<float>::Conjugated(Q);

cout << MatrixUtils::AreEqualMatrices(view, Q * R) << endl;
cout << MatrixUtils::AreEqualMatrices(I, Matrix<float>::Identity(2)) << endl;
```

Больше примеров работы с библиотекой можно найти в [тестах](cp-linear-kit/tests) в папке `tests`.

## 🏗️ Подключение библиотеки

---

Исходный код библиотеки основа исключительно на файлах формата `.h`, находящихся в папках внутри `src`, потому для использования библиотеки потребуется при подключении конкретных файлов указывать явный путь, например:

```cpp
#include "cp-linear-kit/src/types/matrix.h"

using namespace std;
using namespace LinearKit;

auto m1 = Matrix<>::Identity(3);
auto m2 = Matrix<>::Diagonal(Matrix<>({{1}, {2}, {3}}));
auto m3 = Matrix<>({{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}) * 2;

cout << m2 * m3 + m1 << endl;
```

## ✅ Тестирование

---

Для корректной работы библиотеки реализованы тесты в папке `tests`. Для тестовой сборки потребуется `CMake`, начиная с версии `3.26`. Название таргетов для тестирования конкретных файлов совпадают с их названиями, а также определён таргет `test_all` для запуска всех тестов. Пример сборки тестов через `CMake`:
```bash
# in cp-linear-kit
mkdir build && cd build

# configure CMake for build folder
cmake ../ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# compile and run tests!
make test_all && ./test_all
```

## ⚙️ О проекте

---

Данная библиотека является курсовым проектом в рамках НИУ ВШЭ на образовательной программе "Прикладная математика и информатика", 2024 год. Работу выполнил Даниил Тимижев, студент 2 курса. Ознакомиться с отчётом можно [здесь](cp-linear-kit/CP_Report.pdf).
