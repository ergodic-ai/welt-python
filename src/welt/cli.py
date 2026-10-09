"""Small explicit command-line login; secrets are never displayed."""
import argparse
import sys
from .client import Client
from .errors import WeltError
from .login import connect


def main(argv=None):
    parser=argparse.ArgumentParser(prog="welt")
    commands=parser.add_subparsers(dest="command",required=True)
    login=commands.add_parser("login",help="Approve workspace access in your browser")
    login.add_argument("--no-browser",action="store_true",help="Print the verification link without opening a local browser")
    login.add_argument("--no-save",action="store_true",help="Validate a connection without saving a local credential")
    login.add_argument("--base-url",help="Explicit HTTPS API origin; defaults to hosted Welt")
    login.add_argument("--timeout",type=float,default=600)
    args=parser.parse_args(argv)
    try:
        with Client(base_url=args.base_url) as client:
            connect(client,timeout=args.timeout,open_browser=not args.no_browser,
                    save=not args.no_save,client_name="Welt CLI")
    except KeyboardInterrupt:
        print("Stopped waiting locally. The browser code will expire; check your existing connection before retrying.",file=sys.stderr)
        return 130
    except (WeltError,ValueError) as error:
        print(str(error),file=sys.stderr)
        return 1
    return 0
