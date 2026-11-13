from glob import glob

from setuptools import setup

pacote = "va_percecao"

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
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Tomás Frade",
    description="Perceção ao vivo com a câmara traseira do ATLASCAR2 (grupo G6)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "percecao = va_percecao.no_percecao:main",
            "reacao = va_percecao.no_reacao:main",
        ],
    },
)
