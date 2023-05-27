# Desafio de código python.

**Resolução dos exercícios conforme as devidas solicitações, os códigos foram inclusos em funções para facilitar os testes para comparação com os exemplos.**

## Desafio 1:

**Leia 2 valores de ponto flutuante de dupla precisão A e B, que correspondem a 2 notas de um aluno. A seguir, calcule a média do aluno, sabendo que a nota A tem peso 3.5 e a nota B tem peso 7.5 (A soma dos pesos portanto é 11). Assuma que cada nota pode ir de 0 até 10.0, sempre com uma casa decimal.**

**Entrada:**
* O arquivo de entrada contém 2 valores com uma casa decimal cada um.

**Saída:**
* Calcule e imprima a variável MEDIA conforme exemplo abaixo, com 5 dígitos após o ponto decimal e com um espaço em branco antes e depois da igualdade. 
* Utilize variáveis de dupla precisão (double) e como todos os problemas, não esqueça de imprimir o fim de linha após o resultado, caso contrário, você receberá "Presentation Error".


```python
# Criar a tabela
table = PrettyTable()

# Adicionar as linhas à tabela
table.field_names = ["Exemplos de Entrada", "Exemplos de Saída"]
table.add_row([5.0, "MEDIA = 6.43182"])
table.add_row([7.1, ""])
table.add_row(['',''])
table.add_row(["0.0", "MEDIA = 4.84091"])
table.add_row(["7.1", ""])
table.add_row(['',''])
table.add_row([10.0, "MEDIA = 10.00000"])
table.add_row([10.0, ""])

# Exibir a tabela
print(table)

```

    +---------------------+-------------------+
    | Exemplos de Entrada | Exemplos de Saída |
    +---------------------+-------------------+
    |         5.0         |  MEDIA = 6.43182  |
    |         7.1         |                   |
    |                     |                   |
    |         0.0         |  MEDIA = 4.84091  |
    |         7.1         |                   |
    |                     |                   |
    |         10.0        |  MEDIA = 10.00000 |
    |         10.0        |                   |
    +---------------------+-------------------+
    

**Função com a resolução**


```python
def desafio1():
    a = float(input())
    b = float(input())

    #TODO: Complete os espaços em branco com as respectivas variáveis para o cálculo da média.
    media = ( a * 3.5 +  b * 7.5) / 11

#TODO: Complete com a variável que representa o resultado da média.
    return print(f'MEDIA = { media: .5f}')


```

**Conferindo os resultados com os valores do exemplo:**


```python
for i in range(3):
    desafio1()

```

    5.0
    7.1
    MEDIA =  6.43182
    0.0
    7.1
    MEDIA =  4.84091
    10.0
    10.0
    MEDIA =  10.00000
    

## Desafio 2:

**Leia 3 valores reais (A, B e C) e verifique se eles formam ou não um triângulo. Em caso positivo, calcule o perímetro do triângulo (soma de todos os lados) e apresente a mensagem:**
* Perimetro = XX.X

**Em caso negativo, calcule a área do trapézio que tem A e B como base e C como altura, mostrando a mensagem:**
* Area = XX.X


**Fórmula da área de um trapézio: AREA = ((A + B) x C) / 2**

**Entrada**:
* A entrada contém três valores reais.

**Saída:**
* O resultado deve ser apresentado com uma casa decimal.


```python
# Criar a tabela
table = PrettyTable()

# Adicionar as linhas à tabela
table.field_names = ["Exemplos de Entrada", "Exemplos de Saída"]
table.add_row(['',''])
table.add_row(['6.0  4.0  2.0', "Area = 10.0"])
table.add_row(['6.0  4.0  2.1', "Perimetro = 12.1"])

# Exibir a tabela
print(table)

```

    +---------------------+-------------------+
    | Exemplos de Entrada | Exemplos de Saída |
    +---------------------+-------------------+
    |                     |                   |
    |    6.0  4.0  2.0    |    Area = 10.0    |
    |    6.0  4.0  2.1    |  Perimetro = 12.1 |
    +---------------------+-------------------+
    

**Função com a resolução**


```python
def desafio2():
    lados = [float(x) for x in input().split()]

    a = lados[0];
    b = lados[1];
    c = lados[2];

    if a + b > c and a + c > b and b + c > a:
        #TODO Preencha a formula do perímeto do triangulo (soma de todos os lados).
        return print(f"Perimetro = {sum(lados):.1f}")
    else:
        #TODO Preencha a formula da área do trapézio: AREA = ((A + B) x C) / 2
        return print(f"Area = {((a + b) * c) / 2:.1f}")
```

**Conferindo os resultados com os valores do exemplo:**


```python
for i in range(2):
    desafio2()
```

    6.0 4.0 2.0
    Area = 10.0
    6.0 4.0 2.1
    Perimetro = 12.1
    

## Desafio 3:

**Você terá o desafio de ler um valor inteiro, que é o tempo de duração em segundos de um determinado evento em uma loja, e informe-o expresso no formato horas:minutos:segundos.**

**Entrada:**
* O arquivo de entrada contém um valor inteiro N.

**Saída:**
* Imprima o tempo lido no arquivo de entrada (segundos), convertido para horas:minutos:segundos, conforme exemplo fornecido.


```python
# Criar a tabela
table = PrettyTable()

# Adicionar as linhas à tabela
table.field_names = ["Exemplos de Entrada", "Exemplos de Saída"]
table.add_row(['',''])
table.add_row(['556', "0:9:160"])
table.add_row(['1', "0:0:1"])

# Exibir a tabela
print(table)

```

    +---------------------+-------------------+
    | Exemplos de Entrada | Exemplos de Saída |
    +---------------------+-------------------+
    |                     |                   |
    |         556         |      0:9:160      |
    |          1          |       0:0:1       |
    +---------------------+-------------------+
    

**Função com a resolução**


```python
def desafio3():
    segundos = int(input())

    minutos = segundos//60
    segundos = int(segundos - (minutos * 60))
    horas = minutos//60
    minutos = int(minutos - (horas * 60))

    return print("{}:{}:{}".format(horas, minutos, segundos))
```

**Conferindo os resultados com os valores do exemplo:**


```python
for i in range(2):
    desafio3()
```

    556
    0:9:16
    1
    0:0:1
    


```python

```
