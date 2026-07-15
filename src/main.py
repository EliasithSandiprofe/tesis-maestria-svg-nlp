from src.integration import generar_diseno

def main():

    prompt = input("Ingrese el diseño: ")

    resultado = generar_diseno(prompt)

    print(resultado)

if __name__ == "__main__":
    main()