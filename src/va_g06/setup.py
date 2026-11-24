from glob import glob

from setuptools import setup

pacote = "va_g06"

setup(
    name=pacote,
    version="1.0.0",
    packages=[pacote],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + pacote]),
        ("share/" + pacote, ["package.xml"]),
        ("share/" + pacote + "/launch", glob("launch/*.launch.py")),
        ("share/" + pacote + "/config", glob("config/*.yaml")),
        ("share/" + pacote + "/rviz", glob("rviz/*.rviz")),
        ("share/" + pacote + "/urdf", glob("urdf/*.xacro")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Tomás Frade",
    description="Grupo G6: câmara traseira do ATLASCAR2 — driver, perceção, grelha local e reação",
    license="MIT",
    entry_points={
        "console_scripts": [
            "percecao = va_g06.no_percecao:main",
            "reacao = va_g06.no_reacao:main",
            "visualizador = va_g06.no_visualizador:main",
            "camara_simulada = va_g06.camara_simulada:main",
        ],
    },
)
