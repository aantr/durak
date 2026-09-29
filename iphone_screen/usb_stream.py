"""Прямое изображение iPhone → Ubuntu по USB, без Mac Bridge.

python -m iphone_screen.usb_stream --check
python -m iphone_screen.usb_stream
python -m iphone_screen.usb_stream --viewer vnc
"""
from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import PackageNotFoundError, version
import os
import subprocess
import sys

PMD_VERSION = "11.19.3"


async def select_usb_device(udid=None):
    from pymobiledevice3.usbmux import select_devices_by_connection_type

    devices = await select_devices_by_connection_type(connection_type="USB")
    serials = sorted({device.serial for device in devices})
    if udid is not None:
        if udid not in serials:
            raise ValueError(f"iPhone {udid} не найден по USB. Доступны: {', '.join(serials) or 'нет'}")
        return udid
    if not serials:
        raise ValueError("Нет устройств по USB. Подключите iPhone кабелем передачи данных, "
                         "разблокируйте его и подтвердите доверие этому компьютеру. "
                         "На Ubuntu должен работать usbmuxd.")
    if len(serials) > 1:
        raise ValueError("Подключено несколько устройств. Укажите --udid: " + ", ".join(serials))
    return serials[0]


async def inspect_services(udid):
    from pymobiledevice3.remote.rsd_tunnel import PreferredRsdTunnel

    async with PreferredRsdTunnel(serial=udid, prefer_native=False) as rsd:
        names = rsd.peer_info.get("Services", {})
        return {
            "ios": rsd.product_version,
            "display": "com.apple.coredevice.displayservice" in names,
            "screenshots": "com.apple.instruments.dtservicehub" in names,
        }


def require_display(info):
    if not info["display"]:
        fallback = (" Для последовательных скриншотов используйте --viewer screenshots."
                    if info["screenshots"] else "")
        raise ValueError(
            f"iOS {info['ios']}: устройство не объявляет com.apple.coredevice.displayservice. "
            "Web и VNC используют эту же службу и сейчас недоступны. "
            "Это не ошибка браузерного кодека; нужна проверка совместимости iOS и подключённого DDI."
            + fallback)


async def show_screenshots(udid):
    import cv2
    import numpy as np
    from pymobiledevice3.remote.rsd_tunnel import PreferredRsdTunnel
    from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
    from pymobiledevice3.services.dvt.instruments.screenshot import Screenshot

    window = "iPhone USB - screenshots"
    created = False
    try:
        async with PreferredRsdTunnel(serial=udid, prefer_native=False) as rsd:
            async with DvtProvider(rsd) as dvt, Screenshot(dvt) as screenshot:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                created = True
                cv2.resizeWindow(window, 460, 900)
                print("Последовательные скриншоты по USB (не видеопоток). Выход: Q/Esc или Ctrl+C.", flush=True)
                while True:
                    data = await asyncio.wait_for(screenshot.get_screenshot(), timeout=10)
                    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Не удалось декодировать скриншот iPhone")
                    cv2.imshow(window, frame)
                    if cv2.waitKeyEx(1) in (27, ord("q"), ord("Q")):
                        return
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        return
    finally:
        if created:
            cv2.destroyWindow(window)


def run_command(arguments, udid):
    env = os.environ.copy()
    # Не наследуем настройки удалённого tunneld/Mac из текущего терминала.
    for key in ("PYMOBILEDEVICE3_TUNNEL", "PYMOBILEDEVICE3_NATIVE", "PYMOBILEDEVICE3_FORCE_TUNNEL"):
        env.pop(key, None)
    env["PYMOBILEDEVICE3_USERSPACE"] = "1"
    env["PYMOBILEDEVICE3_UDID"] = udid
    subprocess.run([sys.executable, "-m", "pymobiledevice3", *arguments], env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--udid", help="USB-идентификатор iPhone, если подключено несколько устройств")
    parser.add_argument("--viewer", choices=("web", "vnc", "screenshots"), default="web")
    parser.add_argument("--port", type=int, help="Локальный порт: по умолчанию 8080 (web) или 5901 (VNC)")
    parser.add_argument("--check", action="store_true", help="Проверить USB, DDI и наличие службы видеопотока")
    args = parser.parse_args(argv)
    port = args.port if args.port is not None else (8080 if args.viewer == "web" else 5901)
    if not 1 <= port <= 65535:
        parser.error("--port должен быть от 1 до 65535")
    try:
        installed = version("pymobiledevice3")
        if installed != PMD_VERSION:
            raise ValueError(f"Скрипт проверен с pymobiledevice3=={PMD_VERSION}, установлен {installed}. "
                             "Используйте отдельное окружение из iphone_screen/USB_STREAM.md.")
        udid = asyncio.run(select_usb_device(args.udid))
        print(f"iPhone USB: {udid}. Подготовка Developer Disk Image...", flush=True)
        run_command(["mounter", "auto-mount"], udid)
        info = asyncio.run(asyncio.wait_for(inspect_services(udid), timeout=30))
        print(f"iOS {info['ios']}; displayservice: {'есть' if info['display'] else 'нет'}; "
              f"DVT: {'есть' if info['screenshots'] else 'нет'}", flush=True)
        if args.viewer == "screenshots" and not args.check:
            if not info["screenshots"]:
                raise ValueError("Служба DVT для скриншотов тоже отсутствует. Проверьте Developer Disk Image.")
            asyncio.run(show_screenshots(udid))
            return 0
        require_display(info)
        display = ["developer", "core-device", "display"]
        if args.check:
            run_command([*display, "get-media-support-info"], udid)
            print("USB и службы захвата доступны. Для проверки изображения запустите без --check.")
            return 0
        if args.viewer == "web":
            print(f"После запуска сервера откройте http://127.0.0.1:{port}/. Остановка: Ctrl+C.", flush=True)
            run_command([*display, "serve-web", "--bind", "127.0.0.1",
                         "--http-port", str(port), "--no-audio"], udid)
        else:
            print(f"Подключите VNC-клиент к 127.0.0.1:{port}. Остановка: Ctrl+C.", flush=True)
            run_command([*display, "serve-vnc", "--bind", "127.0.0.1",
                         "--port", str(port), "--decoder", "av"], udid)
    except (PackageNotFoundError, ModuleNotFoundError) as exc:
        print(f"Не установлены зависимости: {exc}. См. iphone_screen/USB_STREAM.md", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print("Не удалось запустить службу iPhone; причина указана выше. Проверьте доверие, "
              "режим разработчика и разблокировку экрана.", file=sys.stderr)
        return exc.returncode if 0 < exc.returncode < 126 else 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"USB-захват: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
