import argparse


def main(args):
    print(f"Argumento 1: {args.arg1}")
    print(f"Argumento 2: {args.arg2}")
    print(f"Argumento opcional: {args.optional}")


if __name__ == "__main__":
    # Crear el parser de argumentos
    parser = argparse.ArgumentParser(
        description="Ejemplo de manejo de argumentos en Python"
    )

    # Definir los argumentos que acepta el script
    parser.add_argument("arg1", type=str, help="El primer argumento (requerido)")
    parser.add_argument("arg2", type=int, help="El segundo argumento (requerido)")
    parser.add_argument(
        "--optional",
        type=str,
        default="valor_por_defecto",
        help="Un argumento opcional",
    )

    # Parsear los argumentos de línea de comandos
    args = parser.parse_args()

    # Llamar a la función principal con los argumentos
    main(args)
